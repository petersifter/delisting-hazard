"""Point-in-time merge of Compustat onto the CRSP monthly panel.

Compustat's datadate is the fiscal period end. The filing comes later (median
37 days, 95th percentile 90 days), so figures are attached from their report
date, not their period end.

Covariates follow Campbell, Hilscher and Szilagyi (2008): profitability,
leverage and cash scaled by the market value of total assets.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"

# LC and LU are the usable link types (NR and NU, 63% of the table, mean no
# link). P and C are primary links; N and J are secondary share classes and
# would duplicate firm-years.
LINK_TYPES = {"LC", "LU"}
LINK_PRIM = {"P", "C"}

# If rdq is missing or implausible, assume the report came out one quarter
# after the period end.
FALLBACK_LAG = pd.Timedelta(days=90)
MAX_PLAUSIBLE_LAG = pd.Timedelta(days=365)

# Don't carry a report forward more than two years.
MAX_STALENESS = pd.Timedelta(days=730)

CHS = ["nimta", "tlmta", "cashmta", "mb"]


def load_ccm(path: str | Path | None = None) -> pd.DataFrame:
    """CRSP-Compustat link table, primary LC/LU links only."""
    path = Path(path) if path else RAW / "ccm_link.csv.gz"
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.lower()
    df = df.rename(columns={"lpermno": "permno"})

    before = len(df)
    df = df[df["linktype"].isin(LINK_TYPES) & df["linkprim"].isin(LINK_PRIM)]
    print(f"ccm links: {before:,} -> {len(df):,} after LC/LU and primary filter")

    df["linkdt"] = pd.to_datetime(df["linkdt"])
    # Open links have no end date.
    df["linkenddt"] = pd.to_datetime(df["linkenddt"], errors="coerce").fillna(
        pd.Timestamp("2100-01-01")
    )
    return df[["gvkey", "permno", "linkdt", "linkenddt"]].dropna(subset=["permno"])


def load_quarterly(path: str | Path | None = None) -> pd.DataFrame:
    """Compustat quarterly fundamentals with a publication date."""
    path = Path(path) if path else RAW / "compustat_quarterly.csv.gz"
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.lower()
    df["datadate"] = pd.to_datetime(df["datadate"])
    df["rdq"] = pd.to_datetime(df["rdq"], errors="coerce")

    df["public_date"] = publication_date(df["datadate"], df["rdq"])

    # A few gvkey-datadate pairs repeat; keep the later filing.
    dupes = df.duplicated(["gvkey", "datadate"]).sum()
    if dupes:
        print(f"quarterly: dropping {dupes:,} duplicate (gvkey, datadate) rows")
        df = df.sort_values("public_date").drop_duplicates(["gvkey", "datadate"], keep="last")
    return df


def publication_date(datadate: pd.Series, rdq: pd.Series) -> pd.Series:
    """rdq if it is between 0 and 365 days after datadate, else datadate + 90 days.

    rdq is missing for 35% of rows (all of them before 1970), and a few hundred
    rows have rdq before the period end or years after it.
    """
    lag = rdq - datadate
    usable = rdq.notna() & lag.ge(pd.Timedelta(0)) & lag.le(MAX_PLAUSIBLE_LAG)
    print(f"publication date: {100 * usable.mean():.1f}% from rdq, rest lagged 90d")
    return rdq.where(usable, datadate + FALLBACK_LAG)


def attach_permno(fundamentals: pd.DataFrame, ccm: pd.DataFrame) -> pd.DataFrame:
    """Map gvkey to permno within each link's valid dates.

    Tickers get reused across firms, so matching on ticker produces wrong
    matches that look fine.
    """
    merged = fundamentals.merge(ccm, on="gvkey", how="inner")
    valid = merged["datadate"].between(merged["linkdt"], merged["linkenddt"])
    print(f"link window: {len(merged):,} -> {valid.sum():,} rows inside a valid link")
    return merged[valid].drop(columns=["linkdt", "linkenddt"])


def merge_asof_monthly(monthly: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Attach each firm's latest published quarterly report to each month.

    Backward as-of join on publication date, at most MAX_STALENESS old.
    """
    cols = ["atq", "ltq", "seqq", "ceqq", "txditcq", "pstkq", "niq", "cheq"]
    right = (
        fundamentals[["permno", "public_date", *cols]]
        .dropna(subset=["permno", "public_date"])
        .sort_values("public_date")
    )
    right["permno"] = right["permno"].astype(int)

    left = monthly.sort_values("date").copy()
    out = pd.merge_asof(
        left,
        right,
        left_on="date",
        right_on="public_date",
        by="permno",
        direction="backward",
        tolerance=MAX_STALENESS,
    )
    matched = out["ltq"].notna().mean()
    print(f"fundamentals matched to {100 * matched:.1f}% of security-months")
    return out


def chs_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """Campbell-Hilscher-Szilagyi ratios.

        MTA      market equity + total liabilities
        nimta    net income / MTA
        tlmta    total liabilities / MTA
        cashmta  cash and short-term investments / MTA
        mb       market equity / book equity

    The market value in the denominator moves with the share price, so leverage
    rises as a firm's price falls, before any new filing.
    """
    out = df.copy()
    market_equity = out["mthcap"] / 1_000  # CRSP is in $ thousands, Compustat in $ millions

    # Fama-French book equity from the quarterly file, falling back to the
    # annual figure, which has the full preferred-stock chain.
    book = out["seqq"].fillna(out["ceqq"])
    book = book + out["txditcq"].fillna(0) - out["pstkq"].fillna(0)
    if "book_annual" in out:
        book = book.fillna(out["book_annual"])

    mta = market_equity + out["ltq"]
    mta = mta.where(mta > 0)

    out["nimta"] = out["niq"] / mta
    out["tlmta"] = out["ltq"] / mta
    out["cashmta"] = out["cheq"] / mta
    out["mb"] = market_equity / book.where(book > 0)
    return out


# Annual reports are used from six months after fiscal year end, the
# Fama-French convention.
ANNUAL_LAG = pd.Timedelta(days=182)

# Banks, insurers and real estate, where 90% leverage is normal.
FINANCIAL_SIC = (6000, 6999)


def load_annual(path: str | Path | None = None) -> pd.DataFrame:
    """Annual fundamentals, used for book equity when the quarterly file lacks it.

    Preferred stock is taken at redemption value, then liquidating value, then
    carrying value. The quarterly file only has the last of these.
    """
    path = Path(path) if path else RAW / "compustat_annual.csv.gz"
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.lower()
    df["datadate"] = pd.to_datetime(df["datadate"])
    df["public_date"] = df["datadate"] + ANNUAL_LAG

    preferred = df["pstkrv"].fillna(df["pstkl"]).fillna(df["pstk"])
    equity = df["seq"].fillna(df["ceq"])
    df["book_annual"] = equity + df["txditc"].fillna(0) - preferred.fillna(0)
    return df.dropna(subset=["book_annual"])


def merge_annual_book(monthly: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    """Attach the latest published annual book equity to each firm-month."""
    right = (
        annual[["permno", "public_date", "book_annual"]]
        .dropna(subset=["permno", "public_date"])
        .sort_values("public_date")
    )
    right["permno"] = right["permno"].astype(int)
    out = pd.merge_asof(
        monthly.sort_values("date"),
        right,
        left_on="date",
        right_on="public_date",
        by="permno",
        direction="backward",
        tolerance=MAX_STALENESS,
        suffixes=("", "_ann"),
    )
    return out


def drop_financials(df: pd.DataFrame) -> pd.DataFrame:
    """Remove SIC 6000-6999."""
    lo, hi = FINANCIAL_SIC
    keep = ~df["siccd"].between(lo, hi)
    print(
        f"financials: {len(df):,} -> {keep.sum():,} rows ({100 * (1 - keep.mean()):.1f}% dropped)"
    )
    return df[keep]
