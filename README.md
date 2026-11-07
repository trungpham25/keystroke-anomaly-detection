# Your Typing Is a Fingerprint: Keystroke Anomaly Detection

DS5020 (Intro Linear Algebra & Probability), Fall 2026: Final Project, Project 9 (Anomaly Detection).

Each user's typing rhythm is modelled as a multivariate normal distribution N(μ, Σ). A new
typing sample is flagged as an anomaly (someone else typing the password) when its squared
Mahalanobis distance d² = (x − μ)ᵀ Σ⁻¹ (x − μ) is larger than a χ² threshold.

## Repository layout

| Folder | Contents |
|---|---|
| `src/` | Python code: data download, data loading and description, simulation check |
| `data/` | The raw dataset (kept out of Git; see Setup) |
| `figures/` | Plots produced by the scripts and used in the report |
| `docs/` | Overleaf report PDF, project plan (Excel) and iteration write-ups |

## Setup

```bash
git clone https://github.com/trungpham25/keystroke-anomaly-detection.git
cd keystroke-anomaly-detection
pip install -r requirements.txt
python src/download_data.py
```

`download_data.py` saves the dataset to `data/DSL-StrongPasswordData.csv` (4.7 MB). The file
is ignored by Git, so every member downloads their own copy. To download it by hand instead:

```bash
curl -o data/DSL-StrongPasswordData.csv https://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv
```

## Running the code

All commands are run from the repository root.

| Command | What it does |
|---|---|
| `python src/run_all.py` | Runs the three steps below in order |
| `python src/download_data.py` | Downloads the dataset into `data/` if it is missing |
| `python src/load_data.py` | Describes the real data (size, types, missing values, the UD = DD − H dependency). Saves 3 plots to `figures/` and a summary to `docs/data_summary.txt` |
| `python src/simulation.py` | Simulation with known truth (p = 5, seed 0): true values against estimates, the χ² false-alarm rate, heavy-tailed t(3) data and the effect of sample size. Saves 3 plots to `figures/`, and a log and results table to `docs/` |

Following the course rules, the code imports only `numpy`, `pandas`, `matplotlib` and
`scipy.stats`. Library routines are used only to check our own results.

## Data

The project uses the **CMU Keystroke Dynamics Benchmark** (Killourhy & Maxion, 2009),
<https://www.cs.cmu.edu/~keystroke/>.

- 51 subjects each typed the password `.tie5Roanl` 400 times, in 8 sessions on different days.
- That gives 20,400 rows and 34 columns: `subject`, `sessionIndex`, `rep` and 31 timing features in seconds:
  - 11 hold times (`H.*`)
  - 10 keydown→keydown latencies (`DD.*`)
  - 10 keyup→keydown latencies (`UD.*`)
- One row is one typing of the password. One column is one timing feature. One entry is that time in seconds.
- There are no missing values. `UD = DD − H` holds exactly, so the 31 features have rank 21.
