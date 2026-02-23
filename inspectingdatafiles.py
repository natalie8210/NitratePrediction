from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


RAW_DIR = Path("DATA/RAW")          # adjust if your folder differs
OUT_DIR = Path("DATA/INSPECT")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = OUT_DIR / "raw_parquet_summary.csv"
STATE_COUNTS_CSV = OUT_DIR / "raw_parquet_value_str_counts.csv"


def infer_median_interval_seconds(ts: pd.Series) -> float | None:
    """Return median delta between consecutive timestamps in seconds."""
    if ts is None or ts.empty:
        return None
    ts = ts.dropna().sort_values()
    if len(ts) < 2:
        return None
    deltas = ts.diff().dropna().dt.total_seconds()
    if deltas.empty:
        return None
    return float(deltas.median())


def human_interval(seconds: float | None) -> str:
    if seconds is None or np.isnan(seconds):
        return "NA"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f} min"
    if seconds < 86400:
        return f"{seconds/3600:.2f} hr"
    return f"{seconds/86400:.2f} d"


def inspect_one_file(fp: Path, print_head: bool = False) -> tuple[dict, pd.DataFrame]:
    df = pd.read_parquet(fp)

    for col in ["timestamp", "value_num", "value_str", "value_raw"]:
        if col not in df.columns:
            df[col] = pd.NA

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    n = len(df)
    ts_nonnull = df["timestamp"].notna().sum()
    n_unique_ts = df["timestamp"].nunique(dropna=True)
    dup_ts = int(ts_nonnull - n_unique_ts)

    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()

    interval_sec = infer_median_interval_seconds(df["timestamp"])
    interval_h = human_interval(interval_sec)

    value_num_nonnull = df["value_num"].notna().sum()
    value_num_pct = (value_num_nonnull / n * 100.0) if n else 0.0

    state_counts = (
        df.loc[df["value_num"].isna() & df["value_str"].notna(), "value_str"]
        .astype(str)
        .value_counts()
        .rename_axis("value_str")
        .reset_index(name="count")
    )
    state_counts.insert(0, "series", fp.stem)

    if print_head:
        print("\n" + "=" * 90)
        print(fp.name)
        print(df.head(10))
        print("=" * 90)

    summary = {
        "series": fp.stem,
        "file": fp.name,
        "rows": n,
        "timestamp_nonnull": ts_nonnull,
        "start_utc": ts_min,
        "end_utc": ts_max,
        "unique_timestamps": n_unique_ts,
        "duplicate_timestamps": dup_ts,
        "median_interval": interval_h,
        "median_interval_seconds": interval_sec,
        "value_num_nonnull": value_num_nonnull,
        "value_num_pct": round(value_num_pct, 2),
        "value_str_nonnull": int(df["value_str"].notna().sum()),
    }
    return summary, state_counts


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"RAW_DIR not found: {RAW_DIR.resolve()}")

    files = sorted(RAW_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {RAW_DIR.resolve()}")

    print(f"Found {len(files)} parquet files in {RAW_DIR.resolve()}")

    summaries: list[dict] = []
    all_states: list[pd.DataFrame] = []

    for i, fp in enumerate(files):
        summary, states = inspect_one_file(fp, print_head=(i < 3))
        summaries.append(summary)
        all_states.append(states)

    summary_df = pd.DataFrame(summaries).sort_values(["rows"], ascending=False)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"\nWrote summary CSV: {SUMMARY_CSV.resolve()}")

    states_df = pd.concat(all_states, ignore_index=True)
    states_df = states_df.sort_values(["series", "count"], ascending=[True, False])
    states_df["rank_in_series"] = states_df.groupby("series").cumcount() + 1
    states_df = states_df[states_df["rank_in_series"] <= 10].drop(columns=["rank_in_series"])
    states_df.to_csv(STATE_COUNTS_CSV, index=False)
    print(f"Wrote value_str state counts CSV: {STATE_COUNTS_CSV.resolve()}")

    cols = [
        "series", "rows", "start_utc", "end_utc",
        "median_interval", "value_num_pct", "duplicate_timestamps"
    ]
    print("\nTop-level overview (sorted by rows):")
    print(summary_df[cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()