from __future__ import annotations

import pandas as pd

from delisting.fundamentals import (
    ANNUAL_LAG,
    FALLBACK_LAG,
    chs_covariates,
    drop_financials,
    publication_date,
)


def test_publication_date_uses_rdq():
    datadate = pd.Series(pd.to_datetime(["2003-12-31"]))
    rdq = pd.Series(pd.to_datetime(["2004-02-14"]))
    assert publication_date(datadate, rdq).iloc[0] == pd.Timestamp("2004-02-14")


def test_publication_date_fallback():
    datadate = pd.Series(pd.to_datetime(["2003-12-31"]))
    rdq = pd.Series([pd.NaT])
    assert publication_date(datadate, rdq).iloc[0] == pd.Timestamp("2003-12-31") + FALLBACK_LAG


def test_implausible_rdq_ignored():
    datadate = pd.Series(pd.to_datetime(["2003-12-31", "2003-12-31"]))
    rdq = pd.Series(pd.to_datetime(["2003-06-01", "2010-01-01"]))
    out = publication_date(datadate, rdq)
    assert (out == pd.Timestamp("2003-12-31") + FALLBACK_LAG).all()


def test_annual_lag_longer_than_quarterly():
    assert ANNUAL_LAG > FALLBACK_LAG


def test_book_equity_annual_fallback():
    df = pd.DataFrame(
        {
            "mthcap": [2_000_000.0, 2_000_000.0],
            "seqq": [1000.0, None],
            "ceqq": [None, None],
            "txditcq": [0.0, None],
            "pstkq": [0.0, None],
            "ltq": [500.0, 500.0],
            "niq": [10.0, 10.0],
            "cheq": [50.0, 50.0],
            "book_annual": [900.0, 800.0],
        }
    )
    out = chs_covariates(df)
    assert out["mb"].iloc[0] == 2000.0 / 1000.0
    assert out["mb"].iloc[1] == 2000.0 / 800.0


def test_drop_financials():
    df = pd.DataFrame({"siccd": [3570, 6021, 6999, 7000, 5999]})
    assert drop_financials(df)["siccd"].tolist() == [3570, 7000, 5999]
