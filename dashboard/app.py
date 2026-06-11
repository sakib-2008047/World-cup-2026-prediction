"""World Cup 2026 Prediction Dashboard.

    streamlit run dashboard/app.py

Tabs: Title Race | Stage Probabilities | Matchup Explorer | Methodology.
Reads results/simulation_summary.csv (falls back to running a small batch);
ratings from results/ratings.csv if present, else demo ratings.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import poisson

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from examples.simulate_2026 import DEMO_ELO  # noqa: E402
from src.data.normalize import load_wc2026_teams  # noqa: E402
from src.simulation.monte_carlo import EloScoreSampler, TournamentSimulator  # noqa: E402

RESULTS = PROJECT_ROOT / "results"
SUMMARY_CSV_CANDIDATES = [RESULTS / "simulation_summary.csv",
                          PROJECT_ROOT / "reports" / "simulation_summary.csv"]
RATINGS_CSV = RESULTS / "ratings.csv"
N_SIMS_FALLBACK = 2000

CONFED_COLORS = {
    "UEFA": "#3b6fb6", "CONMEBOL": "#2a9d8f", "CONCACAF": "#e76f51",
    "CAF": "#e9c46a", "AFC": "#9b5de5", "OFC": "#6c757d",
}


# ----------------------------------------------------------- data loading ---
@st.cache_data
def load_ratings() -> dict[str, float]:
    if RATINGS_CSV.exists():
        df = pd.read_csv(RATINGS_CSV)
        return dict(zip(df["team"], df["elo"]))
    return dict(DEMO_ELO)


@st.cache_data
def load_summary() -> tuple[pd.DataFrame, int]:
    for path in SUMMARY_CSV_CANDIDATES:
        if path.exists():
            return pd.read_csv(path, index_col=0), 10_000
    teams = load_wc2026_teams()
    groups = {g: s["team"].tolist() for g, s in teams.groupby("group")}
    sim = TournamentSimulator(load_ratings(), groups)
    return sim.run(n_sims=N_SIMS_FALLBACK, seed=2026), N_SIMS_FALLBACK


@st.cache_data
def team_meta() -> pd.DataFrame:
    return load_wc2026_teams()


# ----------------------------------------------------------------- charts ---
def contenders_chart(summary: pd.DataFrame, confeds: dict, n_sims: int,
                     top_n: int) -> go.Figure:
    top = summary.head(top_n)
    p = top["champion"].values
    se = np.sqrt(p * (1 - p) / n_sims)
    colors = [CONFED_COLORS.get(confeds.get(t, ""), "#888") for t in top.index]
    fig = go.Figure(go.Bar(
        x=p[::-1], y=top.index[::-1], orientation="h",
        marker_color=colors[::-1],
        error_x=dict(type="data", array=1.96 * se[::-1], color="#555",
                     thickness=1.2, width=4),
        text=[f"{v:.1%}" for v in p[::-1]], textposition="outside",
        customdata=np.stack([top["final"].values[::-1],
                             top["semifinal"].values[::-1]], axis=-1),
        hovertemplate=("<b>%{y}</b><br>champion: %{x:.1%}<br>"
                       "final: %{customdata[0]:.1%}<br>"
                       "semis: %{customdata[1]:.1%}<extra></extra>")))
    fig.update_layout(template="plotly_white", height=40 * top_n + 120,
                      xaxis=dict(tickformat=".0%",
                                 range=[0, float(p.max()) * 1.3]),
                      margin=dict(l=10, r=40, t=20, b=10))
    return fig


def matchup_figures(ratings: dict, home: str, away: str, neutral: bool):
    lh, la = EloScoreSampler(ratings).lambdas(home, away, neutral=neutral)
    i = np.arange(7)
    M = np.outer(poisson.pmf(np.arange(11), lh), poisson.pmf(np.arange(11), la))
    M /= M.sum()
    p_home = float(np.tril(M, -1).sum())
    p_draw = float(np.trace(M))
    p_away = float(np.triu(M, 1).sum())

    bar = go.Figure()
    for v, name, color in [(p_home, f"{home} win", "#2a9d8f"),
                           (p_draw, "draw", "#bcbcbc"),
                           (p_away, f"{away} win", "#e76f51")]:
        bar.add_bar(x=[v], y=["outcome"], orientation="h", name=name,
                    marker_color=color, text=f"{v:.0%}",
                    textposition="inside", insidetextanchor="middle")
    bar.update_layout(barmode="stack", template="plotly_white", height=140,
                      xaxis=dict(tickformat=".0%", range=[0, 1]),
                      yaxis=dict(visible=False), showlegend=True,
                      legend=dict(orientation="h", y=-0.5),
                      margin=dict(l=10, r=10, t=10, b=10))

    heat = go.Figure(go.Heatmap(
        z=M[:7, :7], x=[str(x) for x in i], y=[str(x) for x in i],
        colorscale="Viridis", hovertemplate=(
            f"{home} %{{y}} - %{{x}} {away}<br>P=%{{z:.1%}}<extra></extra>")))
    heat.update_layout(template="plotly_white", height=380,
                       xaxis_title=f"{away} goals", yaxis_title=f"{home} goals",
                       margin=dict(l=10, r=10, t=10, b=10))
    return bar, heat, (lh, la, p_home, p_draw, p_away)


# ------------------------------------------------------------------- page ---
def main() -> None:
    st.set_page_config(page_title="World Cup 2026 Predictions",
                       page_icon="⚽", layout="wide")
    ratings = load_ratings()
    summary, n_sims = load_summary()
    meta = team_meta()
    confeds = dict(zip(meta["team"], meta["confederation"]))

    st.title("⚽ FIFA World Cup 2026 — Prediction Engine")
    st.caption(f"Elo ratings + Poisson scorelines + {n_sims:,} Monte Carlo "
               "tournaments of the official 48-team format. "
               "Educational project — not betting advice.")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🏆 Title race", "📊 Stage probabilities", "⚔️ Matchup explorer",
         "🔬 Methodology"])

    with tab1:
        top_n = st.slider("Show top N teams", 5, 20, 10)
        st.plotly_chart(contenders_chart(summary, confeds, n_sims, top_n),
                        use_container_width=True)
        st.caption("Error bars: 95% Monte Carlo confidence interval. "
                   "Colors: confederation.")

    with tab2:
        cols = ["champion", "final", "semifinal", "quarterfinal",
                "knockout_stage"]
        styled = (summary[cols]
                  .rename(columns=lambda c: c.replace("_", " "))
                  .style.format("{:.1%}")
                  .background_gradient(cmap="YlGnBu", vmin=0, vmax=0.5))
        st.dataframe(styled, use_container_width=True, height=600)

    with tab3:
        teams = sorted(ratings)
        c1, c2, c3 = st.columns([2, 2, 1])
        home = c1.selectbox("Team A", teams, index=teams.index("Brazil"))
        away = c2.selectbox("Team B", teams, index=teams.index("Argentina"))
        neutral = c3.toggle("Neutral venue", value=True)
        if home == away:
            st.warning("Pick two different teams.")
        else:
            bar, heat, (lh, la, ph, pd_, pa) = matchup_figures(
                ratings, home, away, neutral)
            m1, m2, m3 = st.columns(3)
            m1.metric(f"{home} expected goals", f"{lh:.2f}")
            m2.metric(f"{away} expected goals", f"{la:.2f}")
            m3.metric("Draw probability", f"{pd_:.0%}")
            st.plotly_chart(bar, use_container_width=True)
            st.plotly_chart(heat, use_container_width=True)

    with tab4:
        st.markdown("""
**Pipeline.** 47k international matches (1872–present) → cleaning with
audited steps → chronological Elo (tournament-weighted K, margin-of-victory
scaling) → leakage-safe features (`shift(1)` long-format pattern) → XGBoost
W/D/L classifier + Poisson scoreline GLM with Dixon–Coles correction →
Monte Carlo over the official 2026 bracket (12 groups, best-8 thirds via
constraint matching equivalent to FIFA Annex C, R32→Final).

**Honesty notes.** Ratings shown may be demo estimates (see repo);
probabilities carry ±~0.7pp simulation noise at 10k runs; penalties modeled
as near coin-flips; fair-play tiebreaker approximated by lots.

Source code, tests (90+), and full methodology: see the GitHub repository.
""")


if __name__ == "__main__":
    main()
