"""Figures for the README."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURES = Path(__file__).resolve().parents[2] / "reports" / "figures"

BLUE, ORANGE, GREEN = "tab:blue", "tab:orange", "tab:green"


def _style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.frameon": False,
            "legend.fontsize": 9,
        }
    )


def _save(fig: plt.Figure, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(FIGURES.parents[1])}")
    return path


def hazard_ratio_figure(failure: pd.Series, merger: pd.Series) -> Path:
    """Failure and merger hazard ratios side by side, one row per covariate."""
    _style()
    order = failure.sort_values().index
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7, 3.8))

    ax.axvline(1.0, color="grey", lw=1, ls="--")
    ax.hlines(y, failure[order], merger[order], color="lightgrey", lw=2, zorder=1)
    ax.scatter(failure[order], y, s=50, color=BLUE, label="failure", zorder=2)
    ax.scatter(merger[order], y, s=50, color=ORANGE, label="merger", zorder=2)

    ax.set_xscale("log")
    ticks = [0.2, 0.5, 1, 2, 3]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_yticks(y, order)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("hazard ratio, 1st vs 99th percentile of covariate (log scale)")
    ax.set_title("Cause-specific Cox models, fitted 1962-2000")
    ax.legend(loc="lower right")
    return _save(fig, "hazard_ratios")


def decile_figure(ew: pd.Series, vw: pd.Series, median: pd.Series) -> Path:
    """Mean and median monthly return by predicted-risk decile."""
    _style()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(1, 11)
    ax.plot(x, vw.values, color=BLUE, marker="o", ms=4, label="value-weighted mean")
    ax.plot(x, ew.values, color=ORANGE, marker="o", ms=4, label="equal-weighted mean")
    ax.plot(x, median.values, color=GREEN, marker="o", ms=4, label="median")
    ax.axhline(0, color="grey", lw=1)
    ax.set_xticks(x)
    ax.set_xlabel("predicted failure-risk decile (10 = riskiest)")
    ax.set_ylabel("monthly return (%)")
    ax.set_title("Return by risk decile, walk-forward scores, 1982-2025")
    ax.legend()
    return _save(fig, "decile_returns")


def equity_figure(vw: pd.DataFrame) -> Path:
    """Growth of $1 in deciles 1 and 10 and the long/short spread, log scale."""
    _style()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    curves = {
        "decile 1 (safest)": vw[1],
        "decile 10 (riskiest)": vw[10],
        "long D1 / short D10": vw[1] - vw[10],
    }
    for (label, r), color in zip(curves.items(), [BLUE, ORANGE, GREEN], strict=True):
        equity = (1 + r).cumprod()
        ax.plot(equity.index, equity.values, color=color, lw=1.5, label=label)

    ax.set_yscale("log")
    ticks = [0.1, 1, 10, 100]
    ax.set_yticks(ticks, [f"${t:g}" for t in ticks])
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_ylabel("growth of $1 (log scale)")
    ax.set_title("Value-weighted deciles, gross of costs")
    ax.legend(loc="upper left")
    return _save(fig, "equity_curves")


def stability_figure(table: pd.DataFrame) -> Path:
    """Failure hazard ratio of each covariate, refitted era by era.

    The y-axis is shared so a small drift doesn't look as large as a big one.
    """
    _style()
    eras = [c for c in table.columns if c != "max/min"]
    fig, axes = plt.subplots(2, 3, figsize=(8.5, 4.4), sharex=True, sharey=True)
    x = np.arange(len(eras))

    for ax, covariate in zip(axes.flat, table.index, strict=False):
        ax.plot(x, table.loc[covariate, eras].values, color=BLUE, marker="o", ms=4)
        ax.axhline(1.0, color="grey", lw=1, ls="--")
        ax.set_yscale("log")
        ax.set_yticks([0.2, 0.5, 1, 2, 3], ["0.2", "0.5", "1", "2", "3"])
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.set_title(covariate, fontsize=10)
        ax.set_xticks(x, [e.replace("-", "-\n") for e in eras], fontsize=8)

    fig.suptitle("Failure hazard ratios refitted by era", fontsize=11)
    fig.supylabel("hazard ratio", fontsize=10)
    fig.tight_layout()
    return _save(fig, "ph_stability")
