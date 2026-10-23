from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW = 12
MIN_PERIODS = 6

FEATURES = ["log_size", "log_price", "vol_12m", "ret_12m", "turnover", "log_age"]

FUNDAMENTALS = ["nimta", "tlmta", "cashmta", "mb"]
FULL = [*FEATURES, *FUNDAMENTALS]
MERGER_FULL = [*FEATURES, *FUNDAMENTALS, "ind_merger_rate", "rel_mb", "rel_size"]


def build_features(monthly: pd.DataFrame) -> pd.DataFrame:
    df = monthly.sort_values(["permno", "date"]).copy()
    g = df.groupby("permno", sort=False)

    df["log_size"] = np.log(df["mthcap"].where(df["mthcap"] > 0))
    df["log_price"] = np.log(df["mthprc"].where(df["mthprc"] > 0))

    df["vol_12m"] = g["ret"].transform(lambda s: s.rolling(WINDOW, min_periods=MIN_PERIODS).std())
    log_ret = np.log1p(df["ret"].where(df["ret"] > -1))
    df["ret_12m"] = np.expm1(
        log_ret.groupby(df["permno"], sort=False).transform(
            lambda s: s.rolling(WINDOW, min_periods=MIN_PERIODS).sum()
        )
    )

    shares = df["shrout"].where(df["shrout"] > 0) * 1_000  # shrout is in thousands
    df["turnover"] = (df["mthvol"] / shares).clip(upper=10)

    df["log_age"] = np.log1p(g.cumcount())

    lagged = [c for c in MERGER_FULL if c in df]
    # lag one month: row t only uses data available at t-1
    df[lagged] = g[lagged].shift(1)
    return df


def counting_process(
    monthly: pd.DataFrame,
    delists: pd.DataFrame,
    outcome: str = "failure",
    features: list[str] | None = None,
) -> pd.DataFrame:
    features = features or FEATURES
    df = build_features(monthly)
    df = df.dropna(subset=features)

    exits = delists.drop_duplicates(subset=["permno"], keep="last").set_index("permno")["outcome"]
    df["exit_outcome"] = df["permno"].map(exits)

    g = df.groupby("permno", sort=False)
    df["stop"] = g.cumcount() + 1
    df["start"] = df["stop"] - 1

    is_last = df["stop"] == g["stop"].transform("max")
    df["event"] = (is_last & df["exit_outcome"].eq(outcome)).astype(int)

    return df[["permno", "date", "start", "stop", "event", *features]].reset_index(drop=True)


def time_split(
    panel: pd.DataFrame, cutoff: str = "2000-12-31"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cut = pd.Timestamp(cutoff)
    return panel[panel["date"] <= cut].copy(), panel[panel["date"] > cut].copy()


MERGER_FEATURES = ["ind_merger_rate", "rel_mb", "rel_size"]


def merger_covariates(monthly: pd.DataFrame, delists: pd.DataFrame) -> pd.DataFrame:
    df = monthly.copy()
    df["sic2"] = (df["siccd"] // 100).astype("Int64")

    merged_firms = set(delists.loc[delists["outcome"] == "merger", "permno"])
    exits = df.groupby("permno")["date"].transform("max")
    df["merged_now"] = (df["date"] == exits) & df["permno"].isin(merged_firms)

    counts = df.groupby(["sic2", "date"], observed=True).agg(
        mergers=("merged_now", "sum"), firms=("permno", "size")
    )
    trailing = (
        counts.groupby(level="sic2", observed=True).rolling(12, min_periods=6).sum().droplevel(0)
    )
    # exclude each month's own mergers from its predictor
    rate = (trailing["mergers"] / trailing["firms"]).groupby(level="sic2").shift(1)
    df["ind_merger_rate"] = df.set_index(["sic2", "date"]).index.map(rate)

    if "log_size" not in df:
        df["log_size"] = np.log(df["mthcap"].where(df["mthcap"] > 0))
    if "mb" not in df:
        raise KeyError("rel_mb needs `mb`; run delisting.fundamentals.chs_covariates first")

    for col, src in [("rel_mb", "mb"), ("rel_size", "log_size")]:
        median = df.groupby(["sic2", "date"], observed=True)[src].transform("median")
        df[col] = df[src] - median

    return df.drop(columns=["merged_now"])
