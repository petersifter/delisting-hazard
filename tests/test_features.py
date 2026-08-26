from __future__ import annotations

import numpy as np
import pandas as pd

from delisting.features import (
    FEATURES,
    build_features,
    counting_process,
    merger_covariates,
    time_split,
)


def _monthly(permnos, months=24, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-31", periods=months, freq="ME")
    return pd.DataFrame(
        {
            "permno": np.repeat(permnos, months),
            "date": list(dates) * len(permnos),
            "ret": rng.normal(0.01, 0.08, months * len(permnos)),
            "mthprc": rng.uniform(5, 50, months * len(permnos)),
            "mthcap": rng.uniform(1e5, 1e7, months * len(permnos)),
            "mthvol": rng.uniform(1e5, 1e6, months * len(permnos)),
            "shrout": rng.uniform(1e3, 1e5, months * len(permnos)),
        }
    )


def _delists(permnos, outcomes):
    return pd.DataFrame({"permno": permnos, "outcome": outcomes})


def test_features_lagged_one_month():
    # Row t must carry the value computed at t-1.
    monthly = _monthly([1])
    out = build_features(monthly)
    expected = np.log(monthly["mthcap"].iloc[0])
    assert np.isclose(out["log_size"].iloc[1], expected)
    assert pd.isna(out["log_size"].iloc[0])


def test_first_month_has_no_features():
    out = build_features(_monthly([1]))
    assert out[FEATURES].iloc[0].isna().all()


def test_event_on_last_month_only():
    panel = counting_process(_monthly([1]), _delists([1], ["failure"]))
    assert panel["event"].sum() == 1
    assert panel["event"].iloc[-1] == 1


def test_merger_is_censored_in_failure_model():
    panel = counting_process(_monthly([1]), _delists([1], ["merger"]))
    assert panel["event"].sum() == 0


def test_outcome_selects_cause():
    monthly = _monthly([1, 2])
    delists = _delists([1, 2], ["failure", "merger"])
    failures = counting_process(monthly, delists, outcome="failure")
    mergers = counting_process(monthly, delists, outcome="merger")
    assert failures.loc[failures["event"] == 1, "permno"].tolist() == [1]
    assert mergers.loc[mergers["event"] == 1, "permno"].tolist() == [2]


def test_survivor_is_censored():
    panel = counting_process(_monthly([1]), _delists([], []))
    assert panel["event"].sum() == 0


def test_intervals_contiguous():
    panel = counting_process(_monthly([1]), _delists([1], ["failure"]))
    assert (panel["stop"] - panel["start"] == 1).all()
    assert panel["start"].is_monotonic_increasing


def test_time_split_no_overlap():
    panel = counting_process(_monthly([1], months=36), _delists([1], ["failure"]))
    train, test = time_split(panel, cutoff="2001-06-30")
    assert train["date"].max() < test["date"].min()
    assert len(train) + len(test) == len(panel)


def test_merger_rate_excludes_current_month():
    monthly = _monthly([1, 2], months=24)
    monthly["siccd"] = 3570
    monthly["mb"] = 1.5
    delists = pd.DataFrame({"permno": [2], "outcome": ["merger"]})
    out = merger_covariates(monthly, delists)

    last_month = out["date"].max()
    assert (
        pd.isna(out.loc[out["date"] == last_month, "ind_merger_rate"]).all()
        or (out.loc[out["date"] == last_month, "ind_merger_rate"] == 0).all()
    )


def test_relative_covariates_demeaned():
    monthly = _monthly([1, 2], months=12)
    monthly["siccd"] = 3570
    monthly["mb"] = [1.0] * 12 + [3.0] * 12
    delists = pd.DataFrame({"permno": [], "outcome": []})
    out = merger_covariates(monthly, delists)
    # Two firms in one industry sit equally far either side of the median.
    by_date = out.groupby("date")["rel_mb"].sum()
    assert np.allclose(by_date.to_numpy(), 0.0)
