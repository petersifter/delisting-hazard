"""Fama-French factor regressions and the deflated Sharpe ratio.

Factors are downloaded from Ken French's data library at runtime and cached in
data/raw/. They are public, so the factor regressions can be rerun without WRDS.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

FRENCH = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FIVE_FACTOR = "F-F_Research_Data_5_Factors_2x3_CSV.zip"
MOMENTUM = "F-F_Momentum_Factor_CSV.zip"

CACHE = Path(__file__).resolve().parents[2] / "data" / "raw"

# The real CSVs are about 50 KB; anything far larger is not a factor file.
MAX_CSV_BYTES = 10_000_000


def _fetch(archive: str) -> pd.DataFrame:
    """Download and parse one monthly Ken French CSV.

    The files open with a text preamble and switch to annual data partway down,
    so only rows whose first field looks like YYYYMM are kept.
    """
    if archive not in (FIVE_FACTOR, MOMENTUM):
        raise ValueError(f"unknown Ken French archive {archive!r}")
    cached = CACHE / archive.replace(".zip", ".csv")
    if not cached.exists():
        with urllib.request.urlopen(FRENCH + archive, timeout=60) as r:
            raw = r.read()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            member = z.infolist()[0]
            if member.file_size > MAX_CSV_BYTES:
                raise ValueError(f"{archive} unpacks to {member.file_size:,} bytes")
            CACHE.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(z.read(member))

    lines = cached.read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip()[:6].isdigit())
    rows = []
    for ln in lines[start:]:
        token = ln.split(",")[0].strip()
        if len(token) != 6 or not token.isdigit():
            break
        rows.append(ln)

    df = pd.read_csv(io.StringIO("\n".join([lines[start - 1], *rows])))
    df.columns = ["date", *[c.strip() for c in df.columns[1:]]]
    df["date"] = pd.to_datetime(df["date"], format="%Y%m") + pd.offsets.MonthEnd(0)
    # Published in percent.
    return df.set_index("date").astype(float) / 100


def load_factors() -> pd.DataFrame:
    """FF5 plus momentum, monthly, indexed on calendar month end."""
    five = _fetch(FIVE_FACTOR)
    mom = _fetch(MOMENTUM)
    mom.columns = ["MOM"]
    return five.join(mom, how="left")


def to_month_end(series: pd.Series) -> pd.Series:
    """Re-date a monthly series to calendar month end.

    CRSP stamps a month with its last trading day (2025-05-30), the French files
    with the calendar month end (2025-05-31). Joined as-is, only months that end
    on a weekday line up.
    """
    out = series.copy()
    out.index = pd.DatetimeIndex(out.index) + pd.offsets.MonthEnd(0)
    if out.index.duplicated().any():
        raise ValueError("more than one observation in a calendar month")
    return out


def alpha(returns: pd.Series, factors: pd.DataFrame, excess: bool = False) -> pd.Series:
    """OLS of a monthly return series on FF5 + momentum, HAC errors with 6 lags.

    A long/short spread is self-financing and is regressed as-is; a long-only
    series has the risk-free rate taken off first (`excess=True`).
    """
    cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]
    df = pd.concat([to_month_end(returns).rename("r"), factors[[*cols, "RF"]]], axis=1)
    df = df.dropna()
    y = df["r"] - df["RF"] if excess else df["r"]
    fit = sm.OLS(y, sm.add_constant(df[cols])).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    return pd.Series(
        {
            "alpha_%_month": 100 * fit.params["const"],
            "alpha_%_year": 100 * ((1 + fit.params["const"]) ** 12 - 1),
            "alpha_t": fit.tvalues["const"],
            **{f"beta_{c}": fit.params[c] for c in cols},
            "r2": fit.rsquared,
            "n_months": int(fit.nobs),
        }
    )


def deflated_sharpe(
    returns: pd.Series, n_trials: int, sr_variance: float | None = None
) -> pd.Series:
    """Deflated Sharpe ratio (Bailey and Lopez de Prado, 2014).

    SR0 = sqrt(V) * ((1 - g) * Z(1 - 1/N) + g * Z(1 - 1/(N e)))
    DSR = Phi((SR - SR0) * sqrt(T - 1) / sqrt(1 - skew * SR + (kurt - 1) / 4 * SR^2))

    SR is the monthly Sharpe, N the number of trials, g the Euler-Mascheroni
    constant and V the variance of the Sharpe estimates across trials. Without
    the individual trial Sharpes, V defaults to 1 / (T - 1), the sampling
    variance of one estimate under the null. Leaving V at 1 would put SR0 on
    the wrong scale and make the threshold far too high.
    """
    r = returns.dropna()
    t = len(r)
    sr = r.mean() / r.std()
    var = 1 / (t - 1) if sr_variance is None else sr_variance

    euler = 0.5772156649
    z1 = stats.norm.ppf(1 - 1 / n_trials)
    z2 = stats.norm.ppf(1 - 1 / (n_trials * np.e))
    sr0 = np.sqrt(var) * ((1 - euler) * z1 + euler * z2)

    skew, kurt = stats.skew(r), stats.kurtosis(r, fisher=False)
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    psr = stats.norm.cdf((sr - sr0) * np.sqrt(t - 1) / denom)

    return pd.Series(
        {
            "sharpe_ann": sr * np.sqrt(12),
            "n_trials": n_trials,
            "threshold_sharpe_ann": sr0 * np.sqrt(12),
            "deflated_sharpe_p": psr,
            "skew": skew,
            "kurtosis": kurt,
        }
    )
