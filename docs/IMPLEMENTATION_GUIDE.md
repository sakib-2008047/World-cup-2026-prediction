# World Cup 2026 Prediction — 18-Phase Implementation Guide

Exact execution plan: Google Colab → GitHub → published portfolio piece.
Every phase: Goal · Files · Colab cells · Commands · Expected output ·
Common mistakes · Validation gate. All source files referenced here already
exist in this repository — your job in Colab is to assemble, run, verify,
and publish them. Do not advance past a phase whose validation gate fails.

**Target structure** (final):

```
world-cup-2026-prediction/
├── data/{raw,processed,external}/
├── notebooks/            # 01_eda, 02_features, 03_models (exploration only)
├── src/{data,features,models,simulation,visualization}/
├── dashboard/app.py
├── examples/             # runnable demos per component
├── tests/                # 93 tests
├── reports/figures/
├── results/              # simulation_summary.csv, metrics.json, ratings.csv
├── docs/IMPLEMENTATION_GUIDE.md (this file)
├── README.md · requirements.txt · .gitignore · LICENSE
```

---

## PHASE 1 — Google Colab setup

**Goal.** A Colab workflow where code lives in GitHub, data persists in
Drive, and every session starts identically.

**Do.** Create the GitHub repo FIRST (empty, with README): name it
`world-cup-2026-prediction`, MIT license. Create a GitHub Personal Access
Token (Settings → Developer settings → Tokens (classic) → scope `repo`).
Create a Kaggle API token (kaggle.com → Settings → Create New Token →
downloads `kaggle.json`).

**Colab cell 1 — session bootstrap** (this cell starts EVERY session):
```python
from google.colab import drive
drive.mount('/content/drive')

import os
GH_USER = "YOUR_USERNAME"
REPO = "world-cup-2026-prediction"
TOKEN = ""  # paste per-session; never commit. Better: store in Colab Secrets (🔑 icon) as GH_TOKEN
if not TOKEN:
    from google.colab import userdata
    TOKEN = userdata.get('GH_TOKEN')

if not os.path.exists(f"/content/{REPO}"):
    !git clone https://{GH_USER}:{TOKEN}@github.com/{GH_USER}/{REPO}.git /content/{REPO}
%cd /content/{REPO}
!git pull
!pip install -q -r requirements.txt 2>/dev/null || pip install -q pandas numpy scipy scikit-learn xgboost matplotlib plotly streamlit pyyaml pyarrow kaggle pytest
```

**Colab cell 2 — Drive-backed data dirs** (raw data survives session resets):
```python
!mkdir -p /content/drive/MyDrive/wc2026/data/raw
!mkdir -p data/processed data/external results reports/figures
!ln -sfn /content/drive/MyDrive/wc2026/data/raw data/raw
```

**Colab cell 3 — Kaggle credentials**:
```python
from google.colab import files
files.upload()  # select kaggle.json
!mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```

**Colab cell 4 — git identity + push helper**:
```python
!git config user.email "you@example.com"
!git config user.name "Your Name"
def push(msg):
    !git add -A && git commit -m "{msg}" && git push
```

**Expected output.** Repo cloned at `/content/world-cup-2026-prediction`,
`data/raw -> Drive` symlink, kaggle.json installed.

**Common mistakes.** (1) Pasting the GitHub token into a committed cell —
use Colab Secrets. (2) Skipping the Drive symlink, then re-downloading 50MB
of data every session. (3) Forgetting `%cd` so files land in `/content`.

**Validation gate.** `!git remote -v` shows your repo; `!ls data/` shows
`raw  processed  external`; `!kaggle datasets list -s football | head -3`
returns rows (credentials work).

---

## PHASE 2 — Dataset collection

**Goal.** All raw datasets in `data/raw/`, with a manifest (source, license,
sha256, timestamp) and the 48-team config verified.

**Files.** `src/data/download.py`, `src/data/normalize.py`,
`data/external/team_name_map.csv`, `data/external/wc2026_teams.csv`,
`data/external/confederations.csv` (all in this repo). In Colab, create
files with `%%writefile src/data/download.py` cells pasting the repo
versions, or simply `git pull` if already pushed.

**Commands.**
```python
!python -m src.data.download          # downloads + validates + manifest
!python -m src.data.download --check  # re-validate without re-downloading
```

**Expected output.**
```
wc2026_teams.csv: 48 teams, 12 groups of 4 - OK
[international_results] ... results.csv: ~48,000 matches, 1872-... to 2026-...
  name-map coverage: all 48 qualified teams resolve OK
[fifa_rankings] ... ~70,000 rows, 1992-12-31 to ...
Manifest written: data/raw/data_manifest.yaml
All required sources acquired and validated.
```

**Common mistakes.** (1) Trusting the Kaggle dataset title "…to 2017" — it
is current. (2) Name-map gaps: if the canary fails (e.g. `['Turkey']`
missing), inspect raw spellings with
`df[df.home_team.str.contains("rk", case=False)].home_team.unique()` and add
one row to `team_name_map.csv`. (3) 403 from Kaggle = accept the dataset's
terms once in the browser.

**Validation gate.** `--check` exits 0; `data_manifest.yaml` exists with 2+
sources; canary message "all 48 qualified teams resolve OK".

---

## PHASE 3 — Data cleaning

**Goal.** `data/processed/matches_clean.parquet` + `fixtures.parquet` +
`cleaning_report.json`, produced by an audited pipeline.

**Files.** `src/data/clean.py`, `tests/test_clean.py`.

**Commands.**
```python
!python -m src.data.clean
!python -m pytest tests/test_clean.py -q
```

**Expected output.** Step-by-step log (`parse_dtypes`, `trim_strings`,
`standardize_names`, `handle_missing`, `remove_duplicates`,
`assign_team_ids`, `validate ... 8 hard checks passed`), then
`Wrote matches_clean.parquet (~47,000 rows)`. Tests: `21 passed`.

**Common mistakes.** Re-running on already-clean data and double-counting
removals (pipeline reads raw each time — fine); editing parquet by hand;
"fixing" the future-fixtures split by imputing 0-0 (never impute scores).

**Validation gate.** All hard checks pass; `cleaning_report.json` rows
in/out reconcile; this notebook check passes:
```python
import pandas as pd
df = pd.read_parquet("data/processed/matches_clean.parquet")
assert df.date.is_monotonic_increasing and not df.duplicated(["date","home_team","away_team"]).any()
```

---

## PHASE 4 — Feature engineering

**Goal.** Leakage-safe feature matrix via the long-format `shift(1)` pattern:
form (5/10), win% (20), goals for/against, GD, opponent strength, WC
experience, continent advantage, FIFA-rank as-of merge.

**Files.** `src/features/build_features.py`, `tests/test_features.py`
(Elo module arrives in Phase 5; create both files now — features import it).

**Colab cell.**
```python
import pandas as pd
from src.data.clean import clean_results
from src.data.download import resolve_files, SOURCES
from src.features.build_features import build_features

matches, fixtures = clean_results()
rankings = pd.read_csv(next(p for p in __import__('pathlib').Path('data/raw/fifa_rankings').glob('*.csv')), parse_dates=['rank_date'])
from src.data.normalize import normalize_names
rankings = normalize_names(rankings.rename(columns={'country_full':'home_team'}), ['home_team']).rename(columns={'home_team':'country_full'})
X = build_features(matches, rankings=rankings)
X.to_parquet("data/processed/match_features.parquet")
print(X.shape, X.filter(like="_diff").columns.tolist())
```

**Expected output.** `(~47000, 42)` and the diff-feature list including
`elo_pre_diff`, `form_pts_last5_diff`, `fifa_rank_diff`.

**Common mistakes.** Computing rolling stats on match-level home/away
columns separately (breaks for teams alternating venues — that's why long
format exists); forgetting rankings name-normalization before the as-of
merge; filling first-match NaN form with 0 ("zero form" is a lie).

**Validation gate.** `pytest tests/test_features.py -q` → `14 passed`,
including `test_no_leakage_flipping_last_result_changes_no_feature` — the
test that matters. Spot-check: pre-1993 rows have NaN `fifa_rank`.

---

## PHASE 5 — Elo rating system

**Goal.** Chronological Elo with tournament-weighted K, margin-of-victory
scaling, home advantage — plus its demo figure and final-ratings export.

**Files.** `src/features/elo.py`, `tests/test_elo.py`, `examples/elo_demo.py`.

**Commands.**
```python
!python -m pytest tests/test_elo.py -q     # 11 passed
!python -m examples.elo_demo               # writes reports/figures/elo_demo.png
```
Export real ratings for the simulator + dashboard:
```python
from src.features.elo import add_elo
rated = add_elo(matches)
ratings = rated.attrs["elo_engine"].ratings
pd.DataFrame({"team": list(ratings), "elo": list(ratings.values())}).to_csv("results/ratings.csv", index=False)
sorted(ratings.items(), key=lambda kv:-kv[1])[:10]
```

**Expected output.** Top-10 dominated by Argentina/Spain/France (order may
vary); figure with logistic curve, trajectories, K-factor histogram.

**Common mistakes.** Updating ratings BEFORE recording the pre-match
snapshot (the classic leak — our engine's ordering makes it impossible, so
don't "refactor" it); starting all teams at 1500 and reading the first
20 years of ratings as meaningful (burn-in: ignore pre-1900 ratings).

**Validation gate.** Sanity vs eloratings.net: your top-10 shares ≥7 teams
with theirs. If wildly off → suspect the name map, not the math.

---

## PHASE 6 — Model training

**Goal.** Logistic / RandomForest / XGBoost on W/D/L with TimeSeriesSplit
tuning; train strictly pre-2018-06, test 2018→present.

**Files.** `src/models/train.py`, `tests/test_models.py`.

**Colab cell.**
```python
import pandas as pd
from src.models.train import run_experiment, save_experiment
X = pd.read_parquet("data/processed/match_features.parquet")
exp = run_experiment(X, test_from="2018-06-01", n_iter=25)
print(exp["results"]); save_experiment(exp)
```

**Expected output.** A table where every model beats `baseline_priors` on
log loss (real-data log loss typically ~0.95–1.02 vs baseline ~1.05–1.09;
accuracy ~52–56%). `models/best_model.joblib` + `metrics.json` written —
move metrics to `results/`.

**Common mistakes.** Random `train_test_split` (cardinal sin — features
encode history); tuning against the test window; comparing models on
accuracy (the simulator consumes probabilities — log loss is primary).

**Validation gate.** `pytest tests/test_models.py -q` → 6 passed; every
model's test log loss < baseline; XGBoost top importance is Elo-related.

---

## PHASE 7 — Model evaluation

**Goal.** The backtest story: 2018 + 2022 World Cup matches only, with a
calibration curve — your report's centerpiece.

**Do (notebook `notebooks/03_models.ipynb`).**
```python
wc = X[(X.tournament=="FIFA World Cup") & (X.date >= "2018-06-01")]
Xw, yw, feats = __import__('src.models.train', fromlist=['xy']).xy(wc, exp["features"])
proba = exp["models"]["xgboost"].predict_proba(Xw)
from sklearn.metrics import log_loss
from src.models.train import multiclass_brier
print("WC-only log loss:", log_loss(yw, proba, labels=range(3)))
print("WC-only Brier  :", multiclass_brier(yw, proba))
# calibration: bin max-probability vs empirical accuracy, plot reliability diagram
```

**Expected output.** WC-only metrics slightly worse than the full test
window (World Cups are higher-variance) — that's normal and worth saying.

**Common mistakes.** Cherry-picking the better of the two tournaments;
reporting accuracy without the draw-rate context (~25% of WC group matches
draw — a no-draw predictor caps near 75% on the other two classes).

**Validation gate.** Reliability diagram roughly tracks the diagonal; both
backtest numbers + calibration figure saved into `reports/figures/`.

---

## PHASE 8 — Poisson goal model

**Goal.** Scoreline engine: two Poisson GLMs + Dixon–Coles correction.

**Files.** `src/models/poisson_model.py`, `tests/test_poisson.py`,
`examples/poisson_demo.py`.

**Commands.**
```python
!python -m pytest tests/test_poisson.py -q   # 11 passed
!python -m examples.poisson_demo             # heatmap figure + rho printed
```
Then fit on REAL features and report rho:
```python
from src.models.poisson_model import PoissonGoalModel
train = X[X.date < "2018-06-01"].dropna(subset=["home_avg_goals_scored"])
pm = PoissonGoalModel().fit(train)
print("Dixon-Coles rho:", pm.rho)            # expect small negative, ~ -0.02..-0.08
```

**Common mistakes.** Truncating the score matrix at 5 goals (renormalize at
10+); fitting rho jointly with the GLMs (hold GLMs fixed — only 4 cells
depend on rho); forgetting the matrix is REGULATION time only.

**Validation gate.** rho negative; matrix sums to 1; W/D/L from the Poisson
model correlates with the classifier's on a sample of matches.

---

## PHASE 9 — Monte Carlo simulation engine

**Goal.** 10,000 tournaments of the verified 2026 bracket.

**Files.** `src/simulation/bracket_2026.py`, `src/simulation/monte_carlo.py`,
`src/simulation/tournament.py` (OO + result-locking),
`tests/test_simulation.py`, `tests/test_tournament.py`.

**Commands.**
```python
!python -m pytest tests/test_simulation.py tests/test_tournament.py -q  # 16 passed
```

**Common mistakes.** Sampling W/D/L instead of scorelines (goal difference
decides group ties AND the best-thirds ranking); hardcoding a guessed R32
bracket (ours is verified vs the FIFA schedule); forgetting hosts'
non-neutral matches.

**Validation gate.** The 495-scenario test passes (every third-place
combination admits a valid slot assignment); conservation test passes
(1 champion, 1 finalist, 2 SF, 4 QF, 8 R16, 16 R32, 16 group per run).

---

## PHASE 10 — 2026 tournament simulation (the run that ships)

**Goal.** The headline probabilities, from YOUR ratings, reproducibly.

**Do.** Point the runner at real ratings:
```python
import pandas as pd
from src.data.normalize import load_wc2026_teams
from src.simulation.monte_carlo import TournamentSimulator
ratings = dict(pd.read_csv("results/ratings.csv").values)
teams = load_wc2026_teams()
groups = {g: s["team"].tolist() for g, s in teams.groupby("group")}
summary = TournamentSimulator(ratings, groups).run(n_sims=10_000, seed=2026)
summary.to_csv("results/simulation_summary.csv")
summary.head(10)
```

**Expected output.** ~45s runtime on Colab CPU; favorites in the 7–14%
champion range; hosts visibly boosted in `knockout_stage`.

**Common mistakes.** Quoting 12.43% (false precision — ±0.7pp noise at 10k);
changing the seed between "the run you publish" and "the CSV you commit".

**Validation gate.** `champion` column sums to 1.000; `final` to 2.000;
top team's probability in a plausible 8–16% band; CSV committed and the
commit tagged: `!git tag v1.0-pre-tournament && git push --tags`
(timestamped proof your predictions preceded results — with the tournament
underway, lock REMAINING-stage predictions and say so honestly).

---

## PHASE 11 — Visualizations

**Files.** `src/visualization/plots.py`, `examples/make_figures.py`.

**Commands.** `!python -m examples.make_figures`

**Expected output.** Five files in `reports/figures/`: `elo_rankings.png`,
`feature_importance.png`, `stage_heatmap.png`, `top_contenders.png` (+ the
two interactive `.html`).

**Common mistakes.** Committing only HTML (README needs PNG); regenerating
figures from a different simulation run than the committed CSV.

**Validation gate.** Open each PNG; every figure carries its source note
and N; contenders chart shows error bars.

---

## PHASE 12 — Streamlit dashboard

**Files.** `dashboard/app.py` (in repo).

**Colab test** (tunnel for preview):
```python
!streamlit run dashboard/app.py --server.headless true --server.port 8501 &>/tmp/st.log &
!sleep 8 && curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8501/healthz   # expect 200
# optional interactive preview:
!npx --yes localtunnel --port 8501
```

**Deploy.** share.streamlit.io → New app → your repo → main branch →
`dashboard/app.py`. Ensure `requirements.txt` is at repo root and
`results/simulation_summary.csv` + `results/ratings.csv` are committed
(the app reads them; without them it falls back to a slow 2k-sim boot).

**Common mistakes.** Relative paths breaking on Cloud (app uses
`Path(__file__)`-anchored paths — keep it that way); uncommitted results
CSV; heavy imports outside `st.cache_data`.

**Validation gate.** Local healthz=200; deployed URL loads all four tabs;
matchup explorer responds < 1s.

---

## PHASE 13 — GitHub repository setup

**Do.**
1. Final structure pass: code under `src/`, demos under `examples/`,
   nothing stray in root except the seven standard files.
2. `LICENSE` (MIT), `.gitignore` (in repo — keeps `data/raw/` out).
3. CI — create `.github/workflows/ci.yml`:
```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - run: pip install -r requirements.txt
      - run: ruff check src tests
      - run: pytest -q
```
4. Open 3 issues and close them with linked commits (e.g. "vectorize
   knockout loop", "permutation importance in report", "post-tournament
   retrospective") — visible project management.
5. Repo About: description + topics
   (`machine-learning, monte-carlo, football, world-cup, xgboost, streamlit`).

**Common mistakes.** One giant "final" commit (history IS the portfolio —
push per phase); committing `kaggle.json` (it's gitignored; if it ever
landed in history, rotate the token).

**Validation gate.** CI green badge; clone-from-scratch on a fresh Colab
runs the Quickstart successfully.

---

## PHASE 14 — README creation

**Files.** `README.md` (full version in repo — customize the placeholders).

Required sections, in order: hero image (top-contenders PNG) → one-line
pitch → live dashboard link → headline results table → architecture diagram
→ "three details I'm proud of" → Quickstart (copy-paste runnable) → repo map
→ evaluation summary → honest limitations → data credits/licenses.

**Common mistakes.** Wall-of-text intro (recruiters give it 30 seconds —
image and results first); claiming betting utility; omitting limitations
(their presence signals seniority, not weakness).

**Validation gate.** A friend can run the Quickstart from the README alone;
all links resolve; image renders on GitHub (relative path, committed file).

---

## PHASE 15 — Portfolio screenshots

**Do.** Capture at ~1600px wide, light theme: (1) dashboard Title-race tab,
(2) Matchup explorer with a marquee pairing, (3) stage heatmap, (4) green CI
+ test count, (5) the README hero as rendered on GitHub. Crop browser chrome.
Make a 10–20s screen recording of the matchup explorer → GIF
(`ffmpeg -i rec.mov -vf "fps=12,scale=900:-1" demo.gif`) → embed in README.

**Validation gate.** GIF < 8MB (GitHub renders it inline); screenshots
legible at LinkedIn's compressed sizes.

---

## PHASE 16 — LinkedIn publication

**Do.** Add to Projects section AND write a post. Post skeleton:

> I built a prediction engine for the first 48-team World Cup 🏆⚽
> 10,000 Monte Carlo tournaments · Elo from scratch over 47k matches since
> 1872 · XGBoost + Poisson scoreline models · the new best-8-thirds bracket
> implemented as a constraint-matching problem (all 495 FIFA Annex C
> scenarios tested) · 93 tests incl. a data-leakage proof.
> My model says: [top-3 with probabilities + chart].
> Live dashboard: [link] · Code: [link]
> What it taught me about [time-series validation / honest uncertainty]: …
> #datascience #machinelearning #worldcup2026

Attach the contenders chart + dashboard GIF. Post while the tournament is
live (now) — and announce the retrospective ("after the final, I'll publish
what the model got right and wrong"), which earns a second post in July.

**Validation gate.** Both links work logged-out; chart readable on mobile.

---

## PHASE 17 — Resume description

Three formats, pick per context:

*One-liner:* Built an end-to-end FIFA World Cup 2026 prediction engine
(Python, XGBoost, Monte Carlo) — 10k tournament simulations of the new
48-team format, deployed as a live Streamlit dashboard; 93 tests.

*Bullets:*
- Engineered leakage-safe ML pipeline over 47k international matches
  (1872–present): custom Elo engine, 18 time-aware features, XGBoost W/D/L
  classifier beating baseline by X% log loss on a 2018–2022 World Cup backtest.
- Implemented the official 48-team 2026 format incl. FIFA Annex C
  third-place allocation as a constraint-matching problem, validated across
  all 495 scenarios; 10,000-run Monte Carlo with quantified CI.
- Shipped reproducible pipeline (manifest-hashed data, 93 pytest tests,
  GitHub Actions CI) and a live Plotly/Streamlit dashboard.

Replace X with your real backtest delta. Every claim must be verifiable in
the repo — interviewers click.

---

## PHASE 18 — Final review checklists

**Deployment workflow (end-to-end):**
`git push` → CI (ruff + pytest) green → Streamlit Cloud auto-redeploys from
main → tag releases (`v1.0-pre-tournament`, later `v2.0-retrospective`) →
after each real matchday: append results via `Tournament.lock_result`,
re-run conditional sims, commit updated `results/`, dashboard updates.

**Portfolio checklist.**
☐ README hero renders ☐ live dashboard link works ☐ Quickstart runs on
fresh machine ☐ CI badge green ☐ 93 tests pass ☐ figures regenerate via one
command ☐ pre-tournament tag pushed ☐ LinkedIn post live ☐ pinned repo.

**Recruiter 30-second review (what they see, in order).**
☐ Title says what it does ☐ image proves it works ☐ results table = concrete
outcome ☐ tests + CI = engineering discipline ☐ limitations section =
maturity ☐ commit history = real work over time ☐ live link = it ships.

**Final submission checklist.**
☐ no secrets in history (`git log -p | grep -i kaggle` empty) ☐ licenses
credited ☐ data/raw gitignored but re-downloadable ☐ seed-pinned published
run matches committed CSV ☐ retrospective issue opened for July ☐ repo
topics set ☐ README claims ⊆ code reality.

---

*Total realistic effort assembling in Colab from these files: 15–25 focused
hours. The tournament is live — Phases 1–12 in one sprint week makes your
pre-knockout predictions tag genuinely meaningful.*
