#!/usr/bin/env python3

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
BUCKET_SIZE_GAMES = 1000   # games per bucket; adjust if you changed it
ROLLING_WINDOW = 50        # buckets for smoothing
CI_Z = 1.96                # ~95% normal CI
TAIL_BUCKETS = 200         # how many final buckets to average

FILES = {
    "MC-Card":  "statistics/win_mccard",
    "MC-Strat": "statistics/win_mcstrat",
    "Q-Card":   "statistics/win_qcard",
    "Q-Strat":  "statistics/win_qstrat",
}

# ----------------------------------------------------------------------
# Loading and preprocessing
# ----------------------------------------------------------------------
def load_agent_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # games_seen: cumulative eval games up to that bucket
    df["games_seen"] = (df["bucket"] + 1) * BUCKET_SIZE_GAMES

    # p = win probability per bucket (0–1); win_rate is percent in file
    df["p"] = df["wins"] / BUCKET_SIZE_GAMES
    df["win_rate_pct"] = df["p"] * 100.0

    # Binomial normal-approximation CI per bucket
    df["se"] = np.sqrt(df["p"] * (1.0 - df["p"]) / BUCKET_SIZE_GAMES)
    df["ci_low_pct"] = (df["p"] - CI_Z * df["se"]) * 100.0
    df["ci_high_pct"] = (df["p"] + CI_Z * df["se"]) * 100.0

    # Rolling mean for smoother learning curves
    df["rolling_win_rate_pct"] = df["win_rate_pct"].rolling(
        window=ROLLING_WINDOW,
        min_periods=1
    ).mean()

    return df

results = {name: load_agent_df(path) for name, path in FILES.items()}

# ----------------------------------------------------------------------
# Per-model plots for each category
# ----------------------------------------------------------------------

def slugify(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")

# 1) Raw win-rate curves (per model)
for name, df in results.items():
    s = slugify(name)
    plt.figure()
    plt.plot(df["bucket"], df["win_rate_pct"], label=name)
    plt.axhline(50, linestyle="--")
    plt.xlabel("Training bucket")
    plt.ylabel("Win rate (%)")
    plt.title(f"{name} – Raw Win Rate per Bucket")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"analysis/raw_{s}.png", dpi=300)
    plt.close()

# 2) Smoothed curves (per model)
for name, df in results.items():
    s = slugify(name)
    plt.figure()
    plt.plot(df["games_seen"], df["rolling_win_rate_pct"], label=name)
    plt.axhline(50, linestyle="--")
    plt.xlabel("Games seen")
    plt.ylabel(f"Rolling mean win rate (%) (window={ROLLING_WINDOW})")
    plt.title(f"{name} – Smoothed Learning Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"analysis/smoothed_{s}.png", dpi=300)
    plt.close()

# 3) Win-rate with CI band (per model)
for name, df in results.items():
    d = df.copy()

    # --- per-bucket probability ---
    p = d["win_rate_pct"] / 100.0   # turn % into probability

    # --- rolling mean & std over buckets ---
    roll_mean_p = p.rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean()
    roll_std_p  = p.rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).std(ddof=1)

    # standard error of the mean for a window of ROLL_WINDOW buckets
    roll_se = roll_std_p / np.sqrt(ROLLING_WINDOW)

    # --- rolling CI in % ---
    d["roll_wr"]      = roll_mean_p * 100
    d["roll_ci_low"]  = (roll_mean_p - CI_Z * roll_se) * 100
    d["roll_ci_high"] = (roll_mean_p + CI_Z * roll_se) * 100

    # drop the prefix part where the window is not full
    m = d["roll_wr"].notna()
    x = d.loc[m, "games_seen"]
    y = d.loc[m, "roll_wr"]
    lo = d.loc[m, "roll_ci_low"]
    hi = d.loc[m, "roll_ci_high"]

    # --- plot rolling WR with rolling CI band ---
    plt.figure()
    plt.plot(x, y, label=name)
    plt.fill_between(x, lo, hi, alpha=0.2)
    plt.axhline(50.0, linestyle="--", linewidth=1)  # baseline

    plt.xlabel("Games seen")
    plt.ylabel("Rolling mean win rate (%)")
    plt.title(f"{name} – Rolling Win Rate ({ROLLING_WINDOW} buckets) with 95% CI")

    # optional zoom
    plt.ylim(47, 53)

    plt.legend()
    plt.tight_layout()
    s = slugify(name)
    plt.savefig(f"analysis/rolling_ci_{s}.png", dpi=300)
    plt.close()


# ----------------------------------------------------------------------
# Final performance bar chart (still all models together)
# ----------------------------------------------------------------------
names = []
means = []
ci_lows = []
ci_highs = []

for name, df in results.items():
    tail = df.tail(TAIL_BUCKETS)
    mean_wr = tail["win_rate_pct"].mean()
    mean_p = tail["p"].mean()

    se_mean = math.sqrt(mean_p * (1 - mean_p) / (BUCKET_SIZE_GAMES * TAIL_BUCKETS))
    ci_low = (mean_p - CI_Z * se_mean) * 100
    ci_high = (mean_p + CI_Z * se_mean) * 100

    names.append(name)
    means.append(mean_wr)
    ci_lows.append(ci_low)
    ci_highs.append(ci_high)

x = np.arange(len(names))
yerr = [np.array(means) - np.array(ci_lows), np.array(ci_highs) - np.array(means)]

plt.figure()
plt.bar(x, means, yerr=yerr)
plt.xticks(x, names)
plt.ylabel("Win rate (%) (final segment)")
plt.title(f"Final Performance (Last {TAIL_BUCKETS} Buckets)")
plt.ylim(47, 53)
plt.tight_layout()
plt.savefig("analysis/final_performance.png", dpi=300)
plt.close()

print(f"Summary over last {TAIL_BUCKETS} buckets:")
for i, name in enumerate(names):
    print(f"{name}: {means[i]:.2f}%   CI ≈ [{ci_lows[i]:.2f}%, {ci_highs[i]:.2f}%]")

