"""Download the CRSP and Compustat tables this project uses to data/raw/.

Each extract is written to parquet and skipped if the file already exists, so a
failed run can be restarted.

    uv run python -m delisting.wrds_extract --all
    uv run python -m delisting.wrds_extract crsp_delist ccm_link
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import wrds

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"

# Ordinary US common stock in the CIZ flags (CIZ replaced SIZ's shrcd/exchcd).
CIZ_UNIVERSE = """
      b.sharetype = 'NS' and b.securitytype = 'EQTY'
  and b.securitysubtype = 'COM' and b.usincflg = 'Y'
  and b.issuertype in ('ACOR', 'CORP')
  and b.primaryexch in ('N', 'A', 'Q')
  and b.conditionaltype = 'RW' and b.tradingstatusflg = 'A'
"""


def _save(df: pd.DataFrame, name: str) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{name}.parquet"
    df.to_parquet(path, index=False)
    print(f"  wrote {path.name}  {len(df):,} rows  {path.stat().st_size / 1e6:.1f} MB")
    return path


def crsp_monthly(db: wrds.Connection) -> pd.DataFrame:
    """CIZ monthly returns joined to the security-info history in force that month.

    Under CIZ the delisting return is already in mthret.
    """
    return db.raw_sql(
        f"""
        select a.permno, a.permco, a.mthcaldt as date,
               a.mthret as ret, a.mthretx as retx, a.mthprc as prc,
               a.mthvol as vol, a.mthcap as mktcap, a.shrout,
               a.mthdelflg, a.mthretflg,
               b.siccd, b.naics, b.ticker, b.issuernm, b.primaryexch,
               b.delactiontype, b.delstatustype, b.delreasontype,
               b.securityenddt
        from crsp.msf_v2 as a
        left join crsp.stksecurityinfohist as b
          on a.permno = b.permno
         and b.secinfostartdt <= a.mthcaldt and a.mthcaldt <= b.secinfoenddt
        where {CIZ_UNIVERSE}
        """,
        date_cols=["date"],
    )


def crsp_delist(db: wrds.Connection) -> pd.DataFrame:
    """Delisting events. delactiontype and delreasontype replace SIZ's dlstcd."""
    return db.raw_sql(
        """
        select permno, permco, delistingdt, delactiontype, delreasontype,
               delistingreturn, delistingprc
        from crsp.stkdelists
        """,
        date_cols=["delistingdt"],
    )


def ccm_link(db: wrds.Connection) -> pd.DataFrame:
    """CRSP-Compustat link table, primary LC/LU links."""
    return db.raw_sql(
        """
        select gvkey, lpermno as permno, linktype, linkprim, linkdt, linkenddt
        from crsp.ccmxpf_lnkhist
        where linktype in ('LU', 'LC') and linkprim in ('P', 'C')
        """,
        date_cols=["linkdt", "linkenddt"],
    )


def compustat_annual(db: wrds.Connection) -> pd.DataFrame:
    """Annual fundamentals, standardized and consolidated."""
    return db.raw_sql(
        """
        select gvkey, datadate, fyear, at, lt, seq, ceq, txditc, pstkl, pstkrv,
               pstk, ib, ni, sale, che
        from comp.funda
        where indfmt = 'INDL' and datafmt = 'STD'
          and popsrc = 'D' and consol = 'C'
        """,
        date_cols=["datadate"],
    )


def compustat_quarterly(db: wrds.Connection) -> pd.DataFrame:
    """Quarterly fundamentals with rdq, the report date."""
    return db.raw_sql(
        """
        select gvkey, datadate, fyearq, fqtr, rdq, atq, ltq, seqq, ceqq,
               ibq, niq, saleq, cheq
        from comp.fundq
        where indfmt = 'INDL' and datafmt = 'STD'
          and popsrc = 'D' and consol = 'C'
        """,
        date_cols=["datadate", "rdq"],
    )


EXTRACTS = {
    "crsp_delist": crsp_delist,
    "ccm_link": ccm_link,
    "compustat_annual": compustat_annual,
    "compustat_quarterly": compustat_quarterly,
    "crsp_monthly": crsp_monthly,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download WRDS tables to data/raw/")
    parser.add_argument("names", nargs="*", help="extracts to run")
    parser.add_argument("--all", action="store_true", help="run every extract")
    parser.add_argument("--force", action="store_true", help="download again if the file exists")
    args = parser.parse_args()

    names = list(EXTRACTS) if args.all else args.names
    if not names:
        parser.error("pass --all or one or more of: " + ", ".join(EXTRACTS))
    unknown = [n for n in names if n not in EXTRACTS]
    if unknown:
        parser.error(f"unknown extract(s): {', '.join(unknown)}")

    db = wrds.Connection()
    try:
        for name in names:
            path = RAW / f"{name}.parquet"
            if path.exists() and not args.force:
                print(f"{name}: already extracted, skipping")
                continue
            print(f"{name}: pulling...")
            t0 = time.time()
            _save(EXTRACTS[name](db), name)
            print(f"  done in {time.time() - t0:.0f}s")
    finally:
        db.close()


if __name__ == "__main__":
    main()
