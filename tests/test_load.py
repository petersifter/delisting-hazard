from __future__ import annotations

import pandas as pd
import pytest

from delisting.load import classify_delisting, load_delists, load_monthly, survival_panel


def _write_monthly(tmp_path, rows):
    df = pd.DataFrame(rows)
    path = tmp_path / "monthly_stock_file.csv.gz"
    df.to_csv(path, index=False, compression="gzip")
    return path


BASE = {
    "issuertype": "CORP",
    "primaryexch": "N",
    "mthretx": 0.01,
    "mthprc": 10.0,
    "mthcap": 1000.0,
    "shrout": 100.0,
    "vwretd": 0.01,
    "sprtrn": 0.01,
}


def test_issuer_and_exchange_filter(tmp_path):
    path = _write_monthly(
        tmp_path,
        [
            {**BASE, "permno": 1, "mthcaldt": "2020-01-31", "mthret": 0.05},
            {**BASE, "permno": 2, "mthcaldt": "2020-01-31", "mthret": 0.05, "issuertype": "ADR"},
            {**BASE, "permno": 3, "mthcaldt": "2020-01-31", "mthret": 0.05, "primaryexch": "Z"},
        ],
    )
    df = load_monthly(path)
    assert df["permno"].tolist() == [1]


def test_duplicate_rows_dropped(tmp_path):
    row = {**BASE, "permno": 1, "mthcaldt": "2020-01-31", "mthret": 0.05}
    path = _write_monthly(tmp_path, [row, row])
    df = load_monthly(path)
    assert len(df) == 1


def test_letter_returns_become_nan(tmp_path):
    path = _write_monthly(
        tmp_path,
        [
            {**BASE, "permno": 1, "mthcaldt": "2020-01-31", "mthret": "C"},
            {**BASE, "permno": 1, "mthcaldt": "2020-02-29", "mthret": 0.02},
        ],
    )
    df = load_monthly(path)
    assert df["ret"].isna().sum() == 1
    assert df["ret"].dtype.kind == "f"


def test_negative_price_is_bidask(tmp_path):
    path = _write_monthly(
        tmp_path, [{**BASE, "permno": 1, "mthcaldt": "2020-01-31", "mthret": 0.0, "mthprc": -8.0}]
    )
    df = load_monthly(path)
    assert bool(df["prc_is_bidask"].iloc[0])
    assert df["mthprc"].iloc[0] == 8.0


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_monthly(tmp_path / "absent.csv.gz")


def test_classify_delisting():
    actions = pd.Series(["GDR", "MER", "GLI", "GEX", "LOS", "???"])
    assert classify_delisting(actions).tolist() == [
        "failure",
        "merger",
        "failure",
        "moved",
        "censored",
        "unknown",
    ]


def _survival_inputs(outcomes):
    permnos = list(range(1, len(outcomes) + 1))
    monthly = pd.DataFrame(
        {
            "permno": [p for p in permnos for _ in range(2)],
            "date": pd.to_datetime(["2020-01-31", "2020-06-30"] * len(permnos)),
        }
    )
    delists = pd.DataFrame(
        {
            "permno": permnos,
            "delist_date": pd.to_datetime(["2020-07-31"] * len(permnos)),
            "outcome": outcomes,
            "delactiontype": outcomes,
            "delreasontype": ["PRICE"] * len(permnos),
            "delret": [-0.5] * len(permnos),
        }
    )
    return monthly, delists


def test_only_failures_are_events():
    monthly, delists = _survival_inputs(["failure", "merger", "moved", "censored"])
    observed = survival_panel(monthly, delists).set_index("permno")["observed"]
    assert observed.loc[1]
    assert not observed.loc[[2, 3, 4]].any()


def test_survivor_right_censored():
    monthly = pd.DataFrame({"permno": [9, 9], "date": pd.to_datetime(["2020-01-31", "2020-06-30"])})
    delists = pd.DataFrame(
        {
            "permno": pd.Series([], dtype=int),
            "delist_date": pd.Series([], dtype="datetime64[ns]"),
            "outcome": pd.Series([], dtype=object),
            "delactiontype": pd.Series([], dtype=object),
            "delreasontype": pd.Series([], dtype=object),
            "delret": pd.Series([], dtype=float),
        }
    )
    panel = survival_panel(monthly, delists)
    assert not panel["observed"].iloc[0]
    assert panel["exit"].iloc[0] == pd.Timestamp("2020-06-30")


def test_delist_date_column(tmp_path):
    path = tmp_path / "delisting_information.csv.gz"
    pd.DataFrame(
        {
            "permno": [1],
            "delistingdt": ["2020-07-31"],
            "deldtprc": [3.2],
            "delnextdt": ["2020-08-15"],
            "delactiontype": ["GDR"],
            "delret": [-0.42],
        }
    ).to_csv(path, index=False, compression="gzip")
    df = load_delists(path)
    assert df["delist_date"].iloc[0] == pd.Timestamp("2020-07-31")
    assert df["outcome"].iloc[0] == "failure"
