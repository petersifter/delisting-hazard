"""Checks on the fitted hazard model: era stability, risk deciles, calibration.

`CoxTimeVaryingFitter` has no Schoenfeld residual test, so stability is checked
by refitting on separate eras and comparing the hazard ratios.
"""

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
    """Hazard ratios refitted era by era, all on the same training bounds.

    The last column is the largest ratio over the smallest. Close to 1 means the
    effect is stable; a large value means the pooled estimate averages over
    relationships that changed.
    """
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
    """Observed failure rate by predicted-risk decile."""
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
    """Attach the linear predictor to every row."""
    out = panel.copy()
    out["risk"] = np.asarray(panel[FEATURES]) @ np.asarray(model.params_)
    return out


def calibration(
    model, train: pd.DataFrame, test: pd.DataFrame, horizon: int = 12, bins: int = 10
) -> pd.DataFrame:
    """Predicted against observed probability of failing within `horizon` months.

    The baseline hazard is one constant, h0 = events / sum(exp(lp)) over the
    training rows (a Breslow estimate with no duration dependence), so

        P(fail within H) = 1 - exp(-H * h0 * exp(lp))

    lifelines' own `baseline_cumulative_hazard_` isn't used because it is built
    on centered covariates and stops at the longest training duration, which
    9.5% of test rows exceed. Dropping duration dependence is a reasonable
    approximation here because log_age is insignificant.
    """
    cols = list(model.params_.index)
    coef = np.asarray(model.params_)

    lp_train = np.asarray(train[cols]) @ coef
    baseline = train["event"].sum() / np.exp(lp_train).sum()

    lp = np.asarray(test[cols]) @ coef
    monthly_hazard = baseline * np.exp(lp)
    predicted = 1 - np.exp(-horizon * monthly_hazard)

    # The event flag is on a firm's last row, so a row counts as a failure within
    # the horizon if the firm failed and its last row is at most `horizon` later.
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
