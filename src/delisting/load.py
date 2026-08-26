"""Load the WRDS downloads into monthly and delisting panels."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"

# Filters the WRDS web query couldn't express (its condition builder has no IN).
# ACOR and CORP are ordinary corporate issuers; N, A and Q are NYSE, NYSE
# American and Nasdaq.
ISSUER_TYPES = {"ACOR", "CORP"}
EXCHANGES = {"N", "A", "Q"}

# Both files are the Annual Update. The delisting file stops on 2025-12-30, one
# day before the last monthly observation, so a firm delisted on 2025-12-31
# would show up as censored, not failed.
SAMPLE_END = pd.Timestamp("2025-12-31")


def _find(pattern: str) -> Path:
    """Find exactly one download in data/raw/ matching `pattern`."""
    matches = sorted(RAW.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no file matching {pattern!r} in {RAW}")
    if len(matches) > 1:
        raise ValueError(f"{pattern!r} matched several files: {[m.name for m in matches]}")
    return matches[0]


def load_monthly(path: str | Path | None = None) -> pd.DataFrame:
    """CIZ monthly stock file, one row per (permno, date).

    Under CIZ the delisting return is already in mthret, so it isn't
    compounded in again here.
    """
    path = Path(path) if path else _find("*[Mm]onthly*.csv.gz")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.lower()
    df = df.rename(columns={"mthcaldt": "date", "mthret": "ret", "mthretx": "retx"})
    df["date"] = pd.to_datetime(df["date"])

    # Missing returns come as blanks or letter codes; make them NaN so the
    # columns stay numeric.
    for col in ("ret", "retx", "mthprc", "mthcap", "shrout", "vwretd", "sprtrn"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df[df["issuertype"].isin(ISSUER_TYPES) & df["primaryexch"].isin(EXCHANGES)]
    df = df[df["date"] <= SAMPLE_END]
    print(f"universe filter: {before:,} -> {len(df):,} rows")

    # A security with two distributions in one month appears twice with
    # otherwise identical rows.
    dupes = df.duplicated(subset=["permno", "date"]).sum()
    if dupes:
        print(f"dropping {dupes:,} duplicate (permno, date) rows from distributions")
        df = df.drop_duplicates(subset=["permno", "date"], keep="first")

    # Negative price = bid-ask midpoint, as in the old SIZ files.
    if "mthprc" in df and (df["mthprc"] < 0).any():
        df["prc_is_bidask"] = df["mthprc"] < 0
        df["mthprc"] = df["mthprc"].abs()

    df = df.sort_values(["permno", "date"]).reset_index(drop=True)
    assert not df.duplicated(subset=["permno", "date"]).any()
    return df


OUTCOMES = {
    "GDR": "failure",  # dropped for failing listing standards
    "GLI": "failure",  # liquidated
    "MER": "merger",  # acquired: a competing risk
    "GEX": "moved",  # changed exchange, still trading
    "LOS": "censored",  # CRSP lost the feed
}


def classify_delisting(action: pd.Series) -> pd.Series:
    """Map delactiontype codes to failure / merger / moved / censored."""
    return action.fillna("").str.upper().map(OUTCOMES).fillna("unknown")


def load_delists(path: str | Path | None = None) -> pd.DataFrame:
    """Delisting events, one row per security that left the tape."""
    path = Path(path) if path else _find("*[Dd]elist*.csv.gz")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.lower()
    df["delist_date"] = pd.to_datetime(df["delistingdt"])
    df["delret"] = pd.to_numeric(df.get("delret"), errors="coerce")
    df["outcome"] = classify_delisting(df["delactiontype"])

    unmatched = df.loc[df["outcome"] == "unknown", "delactiontype"].unique()
    if len(unmatched):
        print(f"unmapped delactiontype values (add to OUTCOMES): {list(unmatched)}")
    print(df["outcome"].value_counts().to_string())

    return df[df["delist_date"] <= SAMPLE_END].sort_values("delist_date")


def survival_panel(monthly: pd.DataFrame, delists: pd.DataFrame) -> pd.DataFrame:
    """One row per security: entry, exit, and whether the exit was a failure.

    Mergers, exchange moves, lost feeds and firms still trading at the end are
    all right-censored at their last month.
    """
    spans = monthly.groupby("permno")["date"].agg(entry="min", exit="max").reset_index()
    events = delists.drop_duplicates(subset=["permno"], keep="last")
    keep = ["permno", "delist_date", "outcome", "delactiontype", "delreasontype", "delret"]
    panel = spans.merge(events[keep], on="permno", how="left")

    panel["observed"] = panel["outcome"].eq("failure")
    panel.loc[panel["observed"], "exit"] = panel.loc[panel["observed"], "delist_date"]
    panel["duration_months"] = ((panel["exit"] - panel["entry"]).dt.days / 30.44).round(1)
    return panel
