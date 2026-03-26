import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from statsmodels.tsa.stattools import acf, pacf, ccf
from statsmodels.tsa.seasonal import STL
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats

# aesthetics
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.size":        11,
})

TARGET    = "IowaRiver_IowaCity_NitrateLevel"
MAX_LAG   = 72          # hours
SEASON    = 24          # daily seasonal period
COLORS    = {
    "ic_nitrate":   "#1a6faf",
    "flow":      "#2ca25f",
    "gauge":     "#8856a7",
    "dam_flow":  "#e6550d",
    "dam_gauge": "#d63b8f",
    "temp":       "#636363",
    "wp_nitrate":  "#31a354",
    "prcp":      "#74add1",
}

# Read in data
df = pd.read_csv("nitrates_ts.csv", parse_dates=["timestamp"])
df = df.set_index("timestamp")
df.index = pd.to_datetime(df.index, utc=True)
df = df.sort_index()

# sqrt-transform precipitation
df["PRCP_sqrt"] = np.sqrt(df["PRCP"])

# short column aliases for readability
COLS = {
    "ic_nitrate":   "IowaRiver_IowaCity_NitrateLevel",
    "flow":      "IowaRiver_IowaCity_Flow",
    "gauge":     "IowaRiver_IowaCity_GaugeHeight",
    "dam_flow":  "IowaRiver_BelowCoralvilleDam_Flow",
    "dam_gauge": "IowaRiver_BelowCoralvilleDam_GaugeHeight",
    "temp":       "WCP_00_TT_091",
    "wp_nitrate":  "WP_WC_Nitrate_River",
    "prcp":      "PRCP_sqrt",
}

PREDICTOR_LABELS = {
    "flow":      "Iowa River Flow (Iowa City)",
    "gauge":     "Iowa River Gauge Height (Iowa City)",
    "dam_flow":  "Coralville Dam Flow",
    "dam_gauge": "Coralville Dam Gauge Height",
    "temp":       "Air Temp",
    "wp_nitrate":  "Nitrate (WP_WC)",
    "prcp":      "Precipitation (sqrt)",
}

print("Dataset loaded.")
print(f"  Rows     : {len(df):,}")
print(f"  Period   : {df.index[0]} → {df.index[-1]}")
print(f"  Columns  : {list(df.columns)}")

# TS overview plots
fig, axes = plt.subplots(len(COLS), 1, figsize=(16, 22), sharex=True)

for ax, (key, col) in zip(axes, COLS.items()):
    ax.plot(df.index, df[col], lw=0.6, color=COLORS[key], alpha=0.85)
    ax.set_ylabel(key, fontsize=9, rotation=0, labelpad=60, ha="right")
    ax.yaxis.set_label_position("left")

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha="right")
fig.tight_layout()

fig.savefig("01_timeseries_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 01_timeseries_overview.png")

# STL decomposition (Trend, Seasonality, Residuals)
nitrate_series = df[TARGET]

for period, label in [(24, "daily"), (24 * 7, "weekly")]:
    try:
        stl = STL(nitrate_series, period=period, robust=True)
        res = stl.fit()
        fig = res.plot()
        fig.set_size_inches(14, 9)
        fig.suptitle(f"STL decomposition {label} ", fontsize=13)
        fig.tight_layout()
        fig.savefig(f"04_stl_{label}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: 04_stl_{label}.png")
    except Exception as e:
        print(f"STL {label} failed: {e}")

# ACF/PACF of target
nit = df[TARGET].values

fig, axes = plt.subplots(2, 1, figsize=(14, 7))

acf_vals, acf_ci  = acf(nit,  nlags=MAX_LAG, alpha=0.05)
pacf_vals, pacf_ci = pacf(nit, nlags=MAX_LAG, alpha=0.05, method="ywmle")

lags = np.arange(len(acf_vals))

# ACF
axes[0].bar(lags, acf_vals, color=COLORS["ic_nitrate"], alpha=0.7, width=0.8)
axes[0].fill_between(lags, acf_ci[:, 0] - acf_vals, acf_ci[:, 1] - acf_vals,
                     alpha=0.2, color="steelblue", label="95% CI")
axes[0].axhline(0, color="black", lw=0.8)
axes[0].set_title("ACF")
axes[0].set_xlabel("Lag (hours)")
axes[0].set_ylabel("Correlation")
axes[0].legend()

# PACF
axes[1].bar(lags, pacf_vals, color=COLORS["ic_nitrate"], alpha=0.7, width=0.8)
axes[1].fill_between(lags, pacf_ci[:, 0] - pacf_vals, pacf_ci[:, 1] - pacf_vals,
                     alpha=0.2, color="steelblue", label="95% CI")
axes[1].axhline(0, color="black", lw=0.8)
axes[1].set_title("PACF")
axes[1].set_xlabel("Lag (hours)")
axes[1].set_ylabel("Partial Correlation")
axes[1].legend()

fig.tight_layout()
fig.savefig("05_acf_pacf_nitrate.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 05_acf_pacf_nitrate.png")

# Ljung-Box test for remaining autocorrelation
lb = acorr_ljungbox(nit, lags=[6, 12, 24, 48, 72], return_df=True)
print("\nLjung-Box test:")
print(lb.to_string())

# ACF/PACF of all predictors
fig, axes = plt.subplots(len(PREDICTOR_LABELS), 2,
                         figsize=(16, 3.5 * len(PREDICTOR_LABELS)))

for i, (key, label) in enumerate(PREDICTOR_LABELS.items()):
    col  = COLS[key]
    data = df[col].dropna().values
    ax_acf, ax_pacf = axes[i]

    try:
        a_vals, a_ci = acf(data,  nlags=MAX_LAG, alpha=0.05)
        p_vals, p_ci = pacf(data, nlags=MAX_LAG, alpha=0.05, method="ywmle")
        lags = np.arange(len(a_vals))

        ax_acf.bar(lags, a_vals, color=COLORS[key], alpha=0.7, width=0.8)
        ax_acf.fill_between(lags, a_ci[:, 0] - a_vals, a_ci[:, 1] - a_vals,
                            alpha=0.2, color=COLORS[key])
        ax_acf.axhline(0, color="black", lw=0.8)
        ax_acf.set_title(f"ACF — {label}", fontsize=10)
        ax_acf.set_xlabel("Lag (hours)")

        ax_pacf.bar(lags, p_vals, color=COLORS[key], alpha=0.7, width=0.8)
        ax_pacf.fill_between(lags, p_ci[:, 0] - p_vals, p_ci[:, 1] - p_vals,
                             alpha=0.2, color=COLORS[key])
        ax_pacf.axhline(0, color="black", lw=0.8)
        ax_pacf.set_title(f"PACF {label}", fontsize=10)
        ax_pacf.set_xlabel("Lag (hours)")

    except Exception as e:
        ax_acf.set_title(f"{label} — failed: {e}", fontsize=9)

fig.tight_layout()
fig.savefig("06_acf_pacf_predictors.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 06_acf_pacf_predictors.png")

# CCF Plots
nit = df[TARGET]

fig, axes = plt.subplots(len(PREDICTOR_LABELS), 1,
                         figsize=(16, 4 * len(PREDICTOR_LABELS)))

CCF_LAGS = 72
lag_summary = {}

for i, (key, label) in enumerate(PREDICTOR_LABELS.items()):
    ax  = axes[i]
    col = COLS[key]

    # align on common index, drop NaNs
    combined = pd.concat([nit, df[col]], axis=1).dropna() # Added .dropna() here
    x = combined[TARGET].values
    y = combined[col].values

    # standardise both series
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()

    # compute CCF for positive and negative lags
    forward  = ccf(x, y, nlags=CCF_LAGS, alpha=None)  # nitrate leads predictor
    backward = ccf(y, x, nlags=CCF_LAGS, alpha=None)  # predictor leads nitrate

    # construct full lag axis: negative = predictor leads nitrate
    fwd  = forward[:CCF_LAGS + 1]
    bwd  = backward[1:CCF_LAGS + 1][::-1]
    full_ccf  = np.concatenate([bwd, fwd])
    full_lags = np.concatenate([-np.arange(len(bwd), 0, -1), np.arange(len(fwd))])

    conf = 1.96 / np.sqrt(len(x))

    colors_bar = [COLORS[key] if abs(v) > conf else "lightgray" for v in full_ccf]
    ax.bar(full_lags, full_ccf, color=colors_bar, width=0.9, alpha=0.85)
    ax.axhline( conf, color="red",  lw=1, ls="--", label=f"±95% CI ({conf:.3f})")
    ax.axhline(-conf, color="red",  lw=1, ls="--")
    ax.axhline(0,     color="black", lw=0.8)
    ax.axvline(0,     color="gray",  lw=0.8, ls=":")
    ax.set_title(f"CCF: Nitrate ↔ {label}", fontsize=11)
    ax.set_xlabel("Lag (hours)  [negative = predictor leads nitrate]")
    ax.set_ylabel("Correlation")
    ax.legend(fontsize=8)

    # find peak lag (predictor leads, so negative lags)
    neg_mask  = full_lags <= 0
    neg_ccf   = full_ccf[neg_mask]
    neg_lags  = full_lags[neg_mask]
    peak_idx  = np.argmax(np.abs(neg_ccf))
    peak_lag  = neg_lags[peak_idx]
    peak_corr = neg_ccf[peak_idx]

    lag_summary[key] = {
        "label":      label,
        "peak_lag_h": int(peak_lag),
        "peak_corr":  round(float(peak_corr), 3),
        "significant": bool(abs(peak_corr) > conf),
    }

    ax.annotate(f"peak lag = {peak_lag}h\nr = {peak_corr:.3f}",
                xy=(peak_lag, peak_corr),
                xytext=(peak_lag + 5, peak_corr + 0.05 * np.sign(peak_corr)),
                fontsize=9, color="black",
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8))

fig.tight_layout()
fig.savefig("07_ccf.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 07_ccf.png")

# Rolling statistics
fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
fig.suptitle("Rolling statistics", fontsize=13)

nit_s = df[TARGET]

axes[0].plot(nit_s.index, nit_s, lw=0.5, color="lightsteelblue", label="Raw")
axes[0].plot(nit_s.rolling(24).mean().index,
             nit_s.rolling(24).mean(),
             lw=1.2, color=COLORS["ic_nitrate"], label="24h rolling mean")
axes[0].plot(nit_s.rolling(168).mean().index,
             nit_s.rolling(168).mean(),
             lw=1.5, color="darkblue", label="168h (7d) rolling mean")
axes[0].set_ylabel("Nitrate (mg/L)")
axes[0].legend(fontsize=9)
axes[0].set_title("Rolling mean")

axes[1].plot(nit_s.rolling(24).std().index,
             nit_s.rolling(24).std(),
             lw=1, color=COLORS["ic_nitrate"], label="24h rolling std")
axes[1].plot(nit_s.rolling(168).std().index,
             nit_s.rolling(168).std(),
             lw=1.5, color="darkblue", label="168h rolling std")
axes[1].set_ylabel("Std dev")
axes[1].legend(fontsize=9)
axes[1].set_title("Rolling standard deviation (variance stability check)")

# rolling correlation: nitrate vs wp_wc nitrate (24h window)
rolling_corr = df[TARGET].rolling(168).corr(df[COLS["wp_nitrate"]])
axes[2].plot(rolling_corr.index, rolling_corr,
             lw=1, color=COLORS["wp_nitrate"])
axes[2].axhline(0, color="black", lw=0.8)
axes[2].set_ylabel("Correlation")
axes[2].set_title("Rolling 168h correlation of nitrate (Iowa City) vs nitrate (WP_WC)")
axes[2].set_xlabel("Date")

axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
axes[2].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=30, ha="right")

fig.tight_layout()
fig.savefig("09_rolling_stats.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 09_rolling_stats.png")

PRCP_THRESH = df["PRCP"].quantile(0.90)   # top 10% = rain event

rain_events  = df[df["PRCP"] > PRCP_THRESH].index
window_after = 72  # hours to observe after rain event

fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=False)
fig.suptitle("Precipitation events and nitrate response", fontsize=13)

# full series with rain events marked
axes[0].plot(df.index, df[TARGET], lw=0.6, color=COLORS["ic_nitrate"], alpha=0.85)
for ev in rain_events:
    axes[0].axvline(ev, color="steelblue", lw=0.3, alpha=0.3)
axes[0].set_title(f"Nitrate with top-10% precipitation events marked "
                  f"(threshold = {PRCP_THRESH:.2f} in)")
axes[0].set_ylabel("Nitrate (mg/L)")

# average nitrate response in the 72h window after rain events
response_matrix = []
for ev in rain_events:
    try:
        window = df[TARGET].loc[ev:ev + pd.Timedelta(hours=window_after)]
        if len(window) >= window_after // 2:
            response_matrix.append(window.values[:window_after + 1])
    except Exception:
        continue

if response_matrix:
    max_len = min(len(r) for r in response_matrix)
    mat     = np.array([r[:max_len] for r in response_matrix])
    mean_r  = mat.mean(axis=0)
    std_r   = mat.std(axis=0)
    hrs     = np.arange(max_len)

    axes[1].plot(hrs, mean_r, color=COLORS["ic_nitrate"], lw=2, label="Mean nitrate")
    axes[1].fill_between(hrs, mean_r - std_r, mean_r + std_r,
                         alpha=0.25, color=COLORS["ic_nitrate"], label="±1 std dev")
    axes[1].axvline(0, color="steelblue", lw=1.5, ls="--", label="Rain event (hour 0)")
    axes[1].set_xlabel("Hours after precipitation event")
    axes[1].set_ylabel("Nitrate (mg/L)")
    axes[1].set_title(f"Average nitrate response after {len(response_matrix)} rain events")
    axes[1].legend(fontsize=9)

fig.tight_layout()
fig.savefig("10_prcp_response.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 10_prcp_response.png")

print("\n" + "=" * 65)
print("LAG DISCOVERY SUMMARY")
print("=" * 65)
print(f"{'Predictor':<35} {'Peak Lag (h)':>12} {'Corr':>8} {'Sig?':>6}")
print("-" * 65)
for key, info in lag_summary.items():
    sig = "YES" if info["significant"] else "no"
    print(f"{info['label']:<35} {info['peak_lag_h']:>12} {info['peak_corr']:>8.3f} {sig:>6}")
print("=" * 65)

# Export summary to CSV
summary_df = pd.DataFrame(lag_summary).T.reset_index(drop=True)
summary_df.to_csv("11_lag_summary.csv", index=False)
print("Table  : 11_lag_summary.csv")
