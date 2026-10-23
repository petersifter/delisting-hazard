"""Cause-specific Cox models for delisting.

Failure and merger are fitted separately, each treating the other as censored.
A merger is usually an acquisition at a premium, so a single pooled model would
fit the same covariates to two opposite outcomes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxTimeVaryingFitter
from lifelines.utils import concordance_index

from delisting.features import FEATURES

# Returns and turnover have very long tails (one stock up 10,000% would dominate
# the likelihood), so covariates are clipped. The bounds come from the training
# years only.
WINSOR = (0.01, 0.99)


def fit_bounds(train: pd.DataFrame, cols: list[str] = FEATURES) -> pd.DataFrame:
    """1st and 99th percentiles of each covariate in the training data."""
    return train[cols].quantile(list(WINSOR))


def apply_bounds(df: pd.DataFrame, bounds: pd.DataFrame) -> pd.DataFrame:
    """Clip each covariate to its training bounds and rescale them to [0, 1].

    x' = (clip(x, p1, p99) - p1) / (p99 - p1)

    A coefficient on x' is the change in log hazard from the training 1st
    percentile of x to its 99th, so exp(coef) compares a firm at the top of the
    range with one at the bottom. It is not a per-standard-deviation effect.
    """
    out = df.copy()
    lo, hi = bounds.iloc[0], bounds.iloc[1]
    out[bounds.columns] = out[bounds.columns].clip(lo, hi, axis=1)
    out[bounds.columns] = (out[bounds.columns] - lo) / (hi - lo).replace(0, np.nan)
    return out


def fit_cox(
    train: pd.DataFrame, penalizer: float = 0.01, features: list[str] | None = None
) -> CoxTimeVaryingFitter:
    """Fit a time-varying Cox model on counting-process rows."""
    features = features or FEATURES
    model = CoxTimeVaryingFitter(penalizer=penalizer)
    model.fit(
        train[["permno", "start", "stop", "event", *features]],
        id_col="permno",
        event_col="event",
        start_col="start",
        stop_col="stop",
        show_progress=False,
    )
    return model


def hazard_ratios(model: CoxTimeVaryingFitter) -> pd.DataFrame:
    """exp(coef) per covariate with its 95% interval and p-value."""
    s = model.summary
    return pd.DataFrame(
        {
            "hazard_ratio": np.exp(s["coef"]),
            "ci_low": np.exp(s["coef lower 95%"]),
            "ci_high": np.exp(s["coef upper 95%"]),
            "p": s["p"],
        }
    ).sort_values("hazard_ratio", ascending=False)


def risk_scores(model: CoxTimeVaryingFitter, panel: pd.DataFrame) -> pd.DataFrame:
    """Linear predictor at each security's last observed month.

    Columns are taken in the model's coefficient order so extra columns in the
    panel can't shift the dot product.
    """
    scored = panel.copy()
    cols = list(model.params_.index)
    scored["risk"] = np.asarray(panel[cols]) @ np.asarray(model.params_)
    return scored.groupby("permno").last().reset_index()


def evaluate(model: CoxTimeVaryingFitter, test: pd.DataFrame) -> float:
    """Harrell's concordance on held-out data (0.5 is random ordering)."""
    last = risk_scores(model, test)
    return concordance_index(last["stop"], -last["risk"], last["event"])


def add_time_interactions(
    panel: pd.DataFrame, features: list[str], kind: str = "duration"
) -> tuple[pd.DataFrame, list[str]]:
    """Add covariate x time columns.

    kind="duration" interacts with log(1 + months since entry), which is the
    proportional hazards assumption itself. kind="calendar" interacts with the
    calendar year, which tests for drift across eras instead (non-stationarity,
    the same thing `subperiod_stability` looks at).

    Returns the panel and the extended feature list.
    """
    out = panel.copy()
    if kind == "duration":
        t = np.log1p(out["stop"])
    elif kind == "calendar":
        t = out["date"].dt.year - out["date"].dt.year.min()
    else:
        raise ValueError(f"kind must be 'duration' or 'calendar', got {kind!r}")

    t = (t - t.mean()) / t.std()
    names = []
    for col in features:
        name = f"{col}_x_t"
        out[name] = out[col] * t
        names.append(name)
    return out, [*features, *names]
