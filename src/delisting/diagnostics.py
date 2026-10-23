from __future__ import annotations

import numpy as np
import pandas as pd

from delisting.features import FEATURES
from delisting.model import apply_bounds, fit_cox, hazard_ratios

ERAS = [
    ("1962-1979", "1962-01-01", "1979-12-31"),
    ("1980-1994", "1980-01-01", "1994-12-31"),
    ("1995-2009", "1995-01-01", "2009-12-31"),
    ("2010-2025", "2010-01-01", "2025-12-31"),
]


def subperiod_stability(panel: pd.DataFrame, bounds: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for name, lo, hi in ERAS:
        era = panel[panel["date"].between(lo, hi)]
        if era["event"].sum() < 100:
            continue
        model = fit_cox(apply_bounds(era, bounds))
        out[name] = hazard_ratios(model)["hazard_ratio"]
    table = pd.DataFrame(out)
    table["max/min"] = table.max(axis=1) / table.min(axis=1)
    return table.sort_values("max/min", ascending=False)


def risk_decile_table(scored: pd.DataFrame) -> pd.DataFrame:
    df = scored.copy()
    df["decile"] = df.groupby("date")["risk"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop") + 1
    )
    table = df.groupby("decile").agg(
        securities=("permno", "size"),
        failures=("event", "sum"),
        mean_risk=("risk", "mean"),
    )
    table["failure_rate_%"] = 100 * table["failures"] / table["securities"]
    return table


def score_panel(model, panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["risk"] = np.asarray(panel[FEATURES]) @ np.asarray(model.params_)
    return out


def calibration(
    model, train: pd.DataFrame, test: pd.DataFrame, horizon: int = 12, bins: int = 10
) -> pd.DataFrame:
    """h0 = events / sum(exp(lp)) on train; P(fail within H) = 1 - exp(-H * h0 * exp(lp))."""
    cols = list(model.params_.index)
    coef = np.asarray(model.params_)

    lp_train = np.asarray(train[cols]) @ coef
    baseline = train["event"].sum() / np.exp(lp_train).sum()

    lp = np.asarray(test[cols]) @ coef
    monthly_hazard = baseline * np.exp(lp)
    predicted = 1 - np.exp(-horizon * monthly_hazard)

    df = test[["permno", "stop", "event"]].copy()
    df["predicted"] = predicted
    last = df.groupby("permno").agg(last_stop=("stop", "max"), failed=("event", "max"))
    df = df.join(last, on="permno")
    df["observed"] = (df["failed"] == 1) & (df["last_stop"] - df["stop"] <= horizon)

    df["bin"] = pd.qcut(df["predicted"].rank(method="first"), bins, labels=False) + 1
    out = df.groupby("bin").agg(
        firm_months=("permno", "size"),
        predicted=("predicted", "mean"),
        observed=("observed", "mean"),
    )
    out[["predicted", "observed"]] *= 100
    out["ratio"] = out["observed"] / out["predicted"].replace(0, np.nan)
    return out
