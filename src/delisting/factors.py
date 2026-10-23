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

MAX_CSV_BYTES = 10_000_000


def _fetch(archive: str) -> pd.DataFrame:
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
    return df.set_index("date").astype(float) / 100  # published in percent


def load_factors() -> pd.DataFrame:
    five = _fetch(FIVE_FACTOR)
    mom = _fetch(MOMENTUM)
    mom.columns = ["MOM"]
    return five.join(mom, how="left")


def to_month_end(series: pd.Series) -> pd.Series:
    """CRSP dates a month by its last trading day, Ken French by the calendar month end."""
    out = series.copy()
    out.index = pd.DatetimeIndex(out.index) + pd.offsets.MonthEnd(0)
    if out.index.duplicated().any():
        raise ValueError("more than one observation in a calendar month")
    return out


def alpha(returns: pd.Series, factors: pd.DataFrame, excess: bool = False) -> pd.Series:
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
    """Bailey and Lopez de Prado (2014), with V = 1 / (T - 1) unless sr_variance is given."""
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
