"""Walk-forward backtest of the failure-hazard score.

Coefficients drift a lot across eras (the log_price hazard ratio moves by a
factor of 3.7), so a model fitted once on 1962-2000 is a poor guide to 2025.
The model is refitted every few years on an expanding window and only scores
months after its training data ends.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from delisting.features import FEATURES
from delisting.model import apply_bounds, fit_bounds, fit_cox

# One-way cost. Optimistic for the top-risk decile, which is mostly sub-$1
# microcaps with wide spreads; roughly fair for the rest.
COST_BPS = 50

REFIT_YEARS = 3
BURN_IN_YEARS = 20


def walk_forward_scores(panel: pd.DataFrame, refit_years: int = REFIT_YEARS) -> pd.DataFrame:
    """Score each month with a model fitted only on earlier months.

    Refits every `refit_years` years on all data before the block and carries
    that model through the block.
    """
    years = sorted(panel["date"].dt.year.unique())
    first = years[0] + BURN_IN_YEARS
    scored = []

    for year in range(first, years[-1] + 1, refit_years):
        train = panel[panel["date"].dt.year < year]
        ahead = panel[panel["date"].dt.year.between(year, year + refit_years - 1)]
        if ahead.empty or train["event"].sum() < 100:
            continue

        bounds = fit_bounds(train)
        model = fit_cox(apply_bounds(train, bounds))

        block = apply_bounds(ahead, bounds)
        block["risk"] = np.asarray(block[FEATURES]) @ np.asarray(model.params_)
        scored.append(block[["permno", "date", "risk", "event"]])
        print(f"  {year}-{year + refit_years - 1}: trained on {len(train):,} rows")

    return pd.concat(scored, ignore_index=True)


def decile_returns(scored: pd.DataFrame, monthly: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Equal-weighted monthly return of each risk decile.

    Covariates on row (permno, t) are lagged to t-1, so the score is known at
    the end of t-1 and is paired with the return over month t.
    """
    df = scored.merge(monthly[["permno", "date", "ret"]], on=["permno", "date"], how="inner")
    df = df.dropna(subset=["ret", "risk"])
    df["decile"] = df.groupby("date")["risk"].transform(
        lambda s: pd.qcut(s.rank(method="first"), n, labels=False, duplicates="drop") + 1
    )
    return df.pivot_table(index="date", columns="decile", values="ret", aggfunc="mean")


def turnover(scored: pd.DataFrame, decile: int, n: int = 10) -> float:
    """Average share of a decile's names replaced each month."""
    df = scored.copy()
    df["decile"] = df.groupby("date")["risk"].transform(
        lambda s: pd.qcut(s.rank(method="first"), n, labels=False, duplicates="drop") + 1
    )
    members = df[df["decile"] == decile].groupby("date")["permno"].apply(set)
    changes = [
        len(cur - prev) / max(len(cur), 1)
        for prev, cur in zip(members.iloc[:-1], members.iloc[1:], strict=True)
    ]
    return float(np.mean(changes))


def performance(returns: pd.Series, cost_per_month: float = 0.0) -> dict[str, float]:
    """Annualized summary of a monthly return series after costs.

    The Sharpe here is compound annual return over annualized volatility, so it
    includes the drag from volatility. `factors.deflated_sharpe` uses the
    arithmetic monthly Sharpe instead.
    """
    net = returns - cost_per_month
    equity = (1 + net).cumprod()
    drawdown = equity / equity.cummax() - 1
    ann_return = (1 + net).prod() ** (12 / len(net)) - 1
    ann_vol = net.std() * np.sqrt(12)
    max_dd = drawdown.min()
    return {
        "ann_return_%": 100 * ann_return,
        "ann_vol_%": 100 * ann_vol,
        "sharpe": ann_return / ann_vol if ann_vol else np.nan,
        "max_drawdown_%": 100 * max_dd,
        "calmar": ann_return / abs(max_dd) if max_dd else np.nan,
        "hit_rate_%": 100 * (net > 0).mean(),
        "months": len(net),
    }
