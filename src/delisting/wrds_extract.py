"""Download the CRSP and Compustat tables this project uses.

Writes gzipped CSVs to data/raw/ under the names delisting.load and
delisting.fundamentals look for, with CRSP's own column names. A file that
already exists is skipped, so a failed run can be restarted.

    uv run python -m delisting.wrds_extract --all
    uv run python -m delisting.wrds_extract crsp_delist ccm_link
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import wrds

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"

# Ordinary US common stock in the CIZ flags. issuertype and primaryexch are
# filtered in delisting.load. Status flags (tradingstatusflg, conditionaltype)
# are deliberately not used: they drop halted and delisting securities.
CIZ_UNIVERSE = """
      b.sharetype = 'NS' and b.securitytype = 'EQTY'
  and b.securitysubtype = 'COM' and b.usincflg = 'Y'
"""

COMPUSTAT_FILTER = "indfmt = 'INDL' and datafmt = 'STD' and popsrc = 'D' and consol = 'C'"


def query(db: wrds.Connection, sql: str) -> pd.DataFrame:
    """Run SQL through the DBAPI connection underneath wrds.

    `db.raw_sql` passes pandas a SQLAlchemy 1.4 connection, which pandas 3 no
    longer accepts, and wrds pins sqlalchemy<2.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
        return pd.read_sql_query(sql, db.connection.connection)


def crsp_monthly(db: wrds.Connection) -> pd.DataFrame:
    """CIZ monthly stock file joined to the security-info history in force that month."""
    return query(
        db,
        f"""
        select a.permno, a.permco, a.mthcaldt, a.mthret, a.mthretx, a.mthprc,
               a.mthvol, a.mthcap, a.shrout, a.mthdelflg, a.mthretflg,
               b.siccd, b.naics, b.ticker, b.issuernm, b.issuertype, b.primaryexch,
               b.delactiontype, b.delstatustype, b.delreasontype, b.securityenddt
        from crsp.msf_v2 as a
        join crsp.stksecurityinfohist as b
          on a.permno = b.permno
         and b.secinfostartdt <= a.mthcaldt and a.mthcaldt <= b.secinfoenddt
        where {CIZ_UNIVERSE}
        """,
    )


def crsp_delist(db: wrds.Connection) -> pd.DataFrame:
    """Delisting events. delactiontype and delreasontype replace SIZ's dlstcd."""
    return query(
        db,
        """
        select permno, permco, delistingdt, delactiontype, delstatustype,
               delreasontype, delret, deldtprc, delnextdt
        from crsp.stkdelists
        """,
    )


def ccm_link(db: wrds.Connection) -> pd.DataFrame:
    """CRSP-Compustat link table, unfiltered; delisting.fundamentals filters it."""
    return query(
        db,
        """
        select gvkey, lpermno, linktype, linkprim, linkdt, linkenddt
        from crsp.ccmxpf_lnkhist
        """,
    )


def compustat_annual(db: wrds.Connection) -> pd.DataFrame:
    """Annual fundamentals in USD, active and inactive firms."""
    return query(
        db,
        f"""
        select gvkey, datadate, fyear, at, lt, seq, ceq, txditc, pstkl, pstkrv,
               pstk, ib, ni, sale, che
        from comp.funda
        where {COMPUSTAT_FILTER} and curcd = 'USD'
        """,
    )


def compustat_quarterly(db: wrds.Connection) -> pd.DataFrame:
    """Quarterly fundamentals in USD, with rdq, the report date."""
    return query(
        db,
        f"""
        select gvkey, datadate, fyearq, fqtr, rdq, atq, ltq, seqq, ceqq,
               txditcq, pstkq, ibq, niq, saleq, cheq
        from comp.fundq
        where {COMPUSTAT_FILTER} and curcdq = 'USD'
        """,
    )


EXTRACTS: dict[str, tuple[Any, str]] = {
    "crsp_delist": (crsp_delist, "crsp_delisting_information.csv.gz"),
    "ccm_link": (ccm_link, "ccm_link.csv.gz"),
    "compustat_annual": (compustat_annual, "compustat_annual.csv.gz"),
    "compustat_quarterly": (compustat_quarterly, "compustat_quarterly.csv.gz"),
    "crsp_monthly": (crsp_monthly, "crsp_monthly_stock_file.csv.gz"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download WRDS tables to data/raw/")
    parser.add_argument("names", nargs="*", help="extracts to run")
    parser.add_argument("--all", action="store_true", help="run every extract")
    parser.add_argument("--force", action="store_true", help="download again if the file exists")
    parser.add_argument("--username", help="WRDS username (password comes from .pgpass)")
    args = parser.parse_args()

    names = list(EXTRACTS) if args.all else args.names
    if not names:
        parser.error("pass --all or one or more of: " + ", ".join(EXTRACTS))
    unknown = [n for n in names if n not in EXTRACTS]
    if unknown:
        parser.error(f"unknown extract(s): {', '.join(unknown)}")

    RAW.mkdir(parents=True, exist_ok=True)
    db = wrds.Connection(wrds_username=args.username) if args.username else wrds.Connection()
    try:
        for name in names:
            pull, filename = EXTRACTS[name]
            path = RAW / filename
            if path.exists() and not args.force:
                print(f"{name}: {filename} exists, skipping")
                continue
            t0 = time.time()
            df = pull(db)
            df.to_csv(path, index=False, compression="gzip")
            print(f"{name}: {len(df):,} rows to {filename} in {time.time() - t0:.0f}s")
    finally:
        db.close()


if __name__ == "__main__":
    main()
