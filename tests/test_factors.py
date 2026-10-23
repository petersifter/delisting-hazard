from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from delisting.factors import _fetch, alpha, to_month_end


def _factors(index: pd.DatetimeIndex, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]
    df = pd.DataFrame(rng.normal(0, 0.04, (len(index), len(cols))), index=index, columns=cols)
    df["RF"] = 0.003
    return df


def test_to_month_end():
    s = pd.Series([0.01, 0.02], index=pd.to_datetime(["2025-05-30", "2025-06-30"]))
    out = to_month_end(s)
    assert list(out.index) == list(pd.to_datetime(["2025-05-31", "2025-06-30"]))


def test_to_month_end_duplicate_month():
    s = pd.Series([0.01, 0.02], index=pd.to_datetime(["2025-05-29", "2025-05-30"]))
    with pytest.raises(ValueError, match="more than one"):
        to_month_end(s)


def test_alpha_with_trading_day_dates():
    # 2020-2024 has many months whose last trading day is not the calendar month end.
    calendar = pd.date_range("2020-01-31", periods=60, freq="ME")
    trading = pd.date_range("2020-01-01", periods=60, freq="BME")
    assert (calendar != trading).sum() > 10

    factors = _factors(calendar)
    returns = pd.Series(0.002 + 1.2 * factors["Mkt-RF"].to_numpy(), index=trading)

    fit = alpha(returns, factors)
    assert fit["n_months"] == 60
    assert fit["beta_Mkt-RF"] == pytest.approx(1.2)


def test_fetch_rejects_unknown_archive():
    with pytest.raises(ValueError, match="unknown Ken French archive"):
        _fetch("../../etc/passwd.zip")
