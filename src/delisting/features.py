"""Hazard covariates and the counting-process panel.

A covariate on the interval (t, t+1] may only use information available at t.
Rolling statistics are trailing and include t, and then every covariate is
lagged one month, so the row for the month a firm failed describes the firm a
month before.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Volatility and momentum use a 12-month window, but six months is accepted so
# young firms, which fail more often than most, stay in the sample.
WINDOW = 12
MIN_PERIODS = 6

FEATURES = ["log_size", "log_price", "vol_12m", "ret_12m", "turnover", "log_age"]

# Campbell-Hilscher-Szilagyi ratios from delisting.fundamentals. They are already
# point-in-time, but they get the same one-month lag as everything else.
FUNDAMENTALS = ["nimta", "tlmta", "cashmta", "mb"]
FULL = [*FEATURES, *FUNDAMENTALS]
MERGER_FULL = [*FEATURES, *FUNDAMENTALS, "ind_merger_rate", "rel_mb", "rel_size"]


def build_features(monthly: pd.DataFrame) -> pd.DataFrame:
    """Attach the six market covariates, lagged one month.

    log_size   log market cap
    log_price  log share price (exchanges delist stocks that stay under $1)
    vol_12m    trailing standard deviation of monthly returns
    ret_12m    trailing 12-month compounded return
    turnover   monthly volume / shares outstanding, capped at 10
    log_age    log(1 + months since the security entered the sample)
    """
    df = monthly.sort_values(["permno", "date"]).copy()
    g = df.groupby("permno", sort=False)

    df["log_size"] = np.log(df["mthcap"].where(df["mthcap"] > 0))
    df["log_price"] = np.log(df["mthprc"].where(df["mthprc"] > 0))

    df["vol_12m"] = g["ret"].transform(lambda s: s.rolling(WINDOW, min_periods=MIN_PERIODS).std())
    # Rolling sum of log returns: same as a rolling product, without a Python
    # call per window.
    log_ret = np.log1p(df["ret"].where(df["ret"] > -1))
    df["ret_12m"] = np.expm1(
        log_ret.groupby(df["permno"], sort=False).transform(
            lambda s: s.rolling(WINDOW, min_periods=MIN_PERIODS).sum()
        )
    )

    # shrout is in thousands, volume in shares.
    shares = df["shrout"].where(df["shrout"] > 0) * 1_000
    df["turnover"] = (df["mthvol"] / shares).clip(upper=10)

    df["log_age"] = np.log1p(g.cumcount())

    # Row t now carries the firm as of t-1.
    lagged = [c for c in MERGER_FULL if c in df]
    df[lagged] = g[lagged].shift(1)
    return df


def counting_process(
    monthly: pd.DataFrame,
    delists: pd.DataFrame,
    outcome: str = "failure",
    features: list[str] | None = None,
) -> pd.DataFrame:
    """One row per firm-month: permno, date, start, stop, event, covariates.

    Only exits of type `outcome` are events. Every other exit, and every firm
    still trading, is censored at its last month.
    """
    features = features or FEATURES
    df = build_features(monthly)
    df = df.dropna(subset=features)

    exits = delists.drop_duplicates(subset=["permno"], keep="last").set_index("permno")["outcome"]
    df["exit_outcome"] = df["permno"].map(exits)

    # Months since the security's first usable row.
    g = df.groupby("permno", sort=False)
    df["stop"] = g.cumcount() + 1
    df["start"] = df["stop"] - 1

    is_last = df["stop"] == g["stop"].transform("max")
    df["event"] = (is_last & df["exit_outcome"].eq(outcome)).astype(int)

    return df[["permno", "date", "start", "stop", "event", *features]].reset_index(drop=True)


def time_split(
    panel: pd.DataFrame, cutoff: str = "2000-12-31"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split on calendar date.

    A random split would put month 47 of a firm in train and month 48 in test,
    and those two rows are nearly identical.
    """
    cut = pd.Timestamp(cutoff)
    return panel[panel["date"] <= cut].copy(), panel[panel["date"] > cut].copy()


MERGER_FEATURES = ["ind_merger_rate", "rel_mb", "rel_size"]


def merger_covariates(monthly: pd.DataFrame, delists: pd.DataFrame) -> pd.DataFrame:
    """Industry-level covariates for the merger model.

    ind_merger_rate  share of same-industry firms acquired over the trailing year
    rel_mb           market-to-book minus the industry-month median
    rel_size         log size minus the industry-month median

    Industry is two-digit SIC. Takeovers come in industry waves and targets tend
    to be cheap and small for their industry, which a firm's own price history
    doesn't capture. The merger rate uses strictly earlier months, and
    build_features adds the usual one-month lag on top.
    """
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
    # Shift so a month's own mergers aren't in its own predictor.
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
