"""Project visualizations - shared style + five chart builders.

Static (matplotlib -> PNG, for README/report):
    plot_elo_rankings        - top-20 lollipop, colored by confederation
    plot_feature_importance  - side-by-side model comparison
    plot_stage_heatmap       - all 48 teams x knockout stages

Interactive (plotly -> HTML, for the Streamlit dashboard):
    plot_win_probability     - W/D/L stacked bars for chosen matchups
    plot_top_contenders      - title-race bars with hover detail + error bars

Style principles applied throughout: one accent palette, direct labeling over
legends where possible, source/N annotations on every figure, no chartjunk.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

FIG_DIR = PROJECT_ROOT / "reports" / "figures"

# ---------------------------------------------------------------- style ----
CONFED_COLORS = {
    "UEFA": "#3b6fb6", "CONMEBOL": "#2a9d8f", "CONCACAF": "#e76f51",
    "CAF": "#e9c46a", "AFC": "#9b5de5", "OFC": "#6c757d", "unknown": "#adb5bd",
}
ACCENT = "#1d3557"
GRID = dict(color="#dddddd", lw=0.6)
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.titleweight": "bold",
})


def _save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _source_note(ax, text: str) -> None:
    ax.annotate(text, xy=(0, -0.13), xycoords="axes fraction",
                fontsize=7.5, color="#777777")


# ----------------------------------------------------- 1. Elo rankings -----
def plot_elo_rankings(ratings: dict[str, float], confeds: dict[str, str],
                      top_n: int = 20, asof: str = "") -> Path:
    """Lollipop chart - cleaner than bars when values share a tight range,
    because the eye reads the dot position, not the bar area."""
    top = sorted(ratings.items(), key=lambda kv: -kv[1])[:top_n][::-1]
    teams = [t for t, _ in top]
    vals = np.array([v for _, v in top])
    colors = [CONFED_COLORS.get(confeds.get(t, "unknown")) for t in teams]

    fig, ax = plt.subplots(figsize=(7.5, 0.34 * top_n + 1.4))
    ax.hlines(range(len(teams)), vals.min() - 40, vals, color="#cccccc", lw=1.2)
    ax.scatter(vals, range(len(teams)), c=colors, s=70, zorder=3)
    for i, (t, v) in enumerate(zip(teams, vals)):
        ax.text(v + 6, i, f"{v:.0f}", va="center", fontsize=8.5)
    ax.set_yticks(range(len(teams)), teams)
    ax.set_xlim(vals.min() - 40, vals.max() + 60)
    ax.set_xlabel("Elo rating")
    ax.set_title(f"Top {top_n} teams by Elo rating{(' — ' + asof) if asof else ''}")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=k)
               for k, c in CONFED_COLORS.items() if k not in ("unknown", "OFC")]
    ax.legend(handles=handles, fontsize=7.5, loc="lower right", frameon=False)
    for x in range(1600, int(vals.max()) + 100, 100):
        ax.axvline(x, **GRID, zorder=0)
    _source_note(ax, "Elo computed from international results, 1872-present "
                     "(tournament-weighted K, margin-of-victory scaled)")
    return _save(fig, "elo_rankings.png")


# ----------------------------------------- 2. Feature importance -----------
def plot_feature_importance(importances: dict[str, pd.Series],
                            top_n: int = 10) -> Path:
    """Side-by-side panels per model. Comparing methods on one canvas makes
    the agreement (or disagreement) the headline - which is the real finding."""
    fig, axes = plt.subplots(1, len(importances),
                             figsize=(5.4 * len(importances), 0.42 * top_n + 1.3),
                             sharey=False)
    fig.subplots_adjust(wspace=0.85)
    if len(importances) == 1:
        axes = [axes]
    for ax, (name, ser) in zip(axes, importances.items()):
        top = ser.head(top_n)[::-1]
        norm = top / ser.max()
        ax.barh(range(len(top)), norm, color=ACCENT, alpha=0.85, height=0.62)
        ax.set_yticks(range(len(top)), [f.replace("_", " ") for f in top.index],
                      fontsize=8.5)
        ax.set_title(name)
        ax.set_xlabel("relative importance")
        ax.set_xlim(0, 1.05)
    fig.suptitle("What drives the predictions — feature importance by model",
                 fontweight="bold", y=1.02)
    _source_note(axes[0], "Importance normalized to each model's max. "
                          "Tree importances are impurity-based; see report for "
                          "permutation check.")
    return _save(fig, "feature_importance.png")


# --------------------------------------------- 3. Win probability ----------
def plot_win_probability(matchups: list[dict], out_name: str =
                         "win_probability.html") -> Path:
    """Interactive stacked W/D/L bars. matchups: [{'home','away','p_home',
    'p_draw','p_away'}]. The 100% stacked form makes draws - the thing
    pundits ignore - visually unavoidable."""
    labels = [f"{m['home']} vs {m['away']}" for m in matchups][::-1]
    p_h = [m["p_home"] for m in matchups][::-1]
    p_d = [m["p_draw"] for m in matchups][::-1]
    p_a = [m["p_away"] for m in matchups][::-1]

    fig = go.Figure()
    for vals, name, color in [(p_h, "home win", "#2a9d8f"),
                              (p_d, "draw", "#bcbcbc"),
                              (p_a, "away win", "#e76f51")]:
        fig.add_bar(y=labels, x=vals, name=name, orientation="h",
                    marker_color=color,
                    text=[f"{v:.0%}" if v >= 0.08 else "" for v in vals],
                    textposition="inside", insidetextanchor="middle",
                    hovertemplate="%{y}<br>" + name + ": %{x:.1%}<extra></extra>")
    fig.update_layout(
        barmode="stack", template="plotly_white",
        title="Match outcome probabilities",
        xaxis=dict(tickformat=".0%", range=[0, 1], title=""),
        legend=dict(orientation="h", y=-0.15),
        height=110 + 52 * len(matchups), margin=dict(l=10, r=10, t=50, b=10),
    )
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / out_name
    fig.write_html(out, include_plotlyjs="cdn")
    return out


# ------------------------------------- 4. Simulation stage heatmap ---------
def plot_stage_heatmap(summary: pd.DataFrame, n_sims: int) -> Path:
    """All 48 teams x 'reached at least' stages. The full-field view that a
    top-10 chart hides: you can see every team's realistic ceiling."""
    cols = ["knockout_stage", "quarterfinal", "semifinal", "final", "champion"]
    data = summary[cols].sort_values("champion", ascending=True)
    fig, ax = plt.subplots(figsize=(6.6, 11.5))
    im = ax.imshow(data.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_yticks(range(len(data)), data.index, fontsize=7.5)
    ax.set_xticks(range(len(cols)),
                  ["R32+", "QF+", "SF+", "Final+", "Champion"], fontsize=9)
    for i in range(len(data)):
        for j, c in enumerate(cols):
            v = data.iloc[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v:.0%}" if v >= 0.095 else f"{v * 100:.1f}",
                        ha="center", va="center", fontsize=6.3,
                        color="white" if v > 0.55 else "#333333")
    ax.set_title(f"Probability of reaching each stage\n"
                 f"({n_sims:,} Monte Carlo tournaments)")
    fig.colorbar(im, shrink=0.5, label="probability")
    _source_note(ax, "2026 format: 12 groups, best-8 thirds, R32 bracket per "
                     "FIFA schedule. Hosts carry Elo home advantage.")
    return _save(fig, "stage_heatmap.png")


# ------------------------------------------- 5. Top-10 contenders ----------
def plot_top_contenders(summary: pd.DataFrame, n_sims: int,
                        out_name: str = "top_contenders.html") -> Path:
    """The headline chart. Includes binomial error bars - the detail that
    tells a reviewer you know what a simulation estimate is."""
    top = summary.head(10)
    p = top["champion"].values
    se = np.sqrt(p * (1 - p) / n_sims)
    colors = [CONFED_COLORS.get(c, "#adb5bd")
              for c in top.get("confederation", pd.Series(["unknown"] * 10))]

    fig = go.Figure(go.Bar(
        x=p[::-1], y=top.index[::-1], orientation="h",
        marker_color=colors[::-1] if "confederation" in top else ACCENT,
        error_x=dict(type="data", array=1.96 * se[::-1], color="#555555",
                     thickness=1.2, width=4),
        text=[f"{v:.1%}" for v in p[::-1]], textposition="outside",
        customdata=np.stack([top["final"].values[::-1],
                             top["semifinal"].values[::-1]], axis=-1),
        hovertemplate=("<b>%{y}</b><br>champion: %{x:.1%}"
                       "<br>reach final: %{customdata[0]:.1%}"
                       "<br>reach semis: %{customdata[1]:.1%}<extra></extra>"),
    ))
    fig.update_layout(
        template="plotly_white",
        title=dict(text=f"Title contenders — P(champion), {n_sims:,} simulations"
                        "<br><sup>error bars: 95% Monte Carlo CI</sup>"),
        xaxis=dict(tickformat=".0%", range=[0, float(p.max()) * 1.25]),
        height=420, margin=dict(l=10, r=30, t=70, b=10),
    )
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / out_name
    fig.write_html(out, include_plotlyjs="cdn")
    return out


def plot_top_contenders_static(summary: pd.DataFrame, n_sims: int) -> Path:
    """Matplotlib twin of the interactive contenders chart - for the README,
    which can embed PNGs but not HTML."""
    top = summary.head(10)
    p = top["champion"].values[::-1]
    se = np.sqrt(p * (1 - p) / n_sims)
    names = top.index[::-1]
    colors = [CONFED_COLORS.get(c, ACCENT)
              for c in top.get("confederation",
                               pd.Series(["unknown"] * 10)).values[::-1]]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.barh(range(10), p, xerr=1.96 * se, color=colors, height=0.62,
            error_kw=dict(lw=1, capsize=3, ecolor="#555555"))
    for i, v in enumerate(p):
        ax.text(v + 1.96 * se[i] + 0.004, i, f"{v:.1%}", va="center", fontsize=9)
    ax.set_yticks(range(10), names)
    ax.set_xlim(0, p.max() * 1.3)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.set_title(f"Title contenders — P(champion), {n_sims:,} simulations")
    _source_note(ax, "Error bars: 95% Monte Carlo confidence interval. "
                     "Colors: confederation.")
    return _save(fig, "top_contenders.png")
