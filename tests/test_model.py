from __future__ import annotations

import pandas as pd
import pytest

from delisting.model import apply_bounds, fit_bounds


def test_apply_bounds_maps_training_percentiles_to_zero_and_one():
    train = pd.DataFrame({"x": range(101)}, dtype=float)
    bounds = fit_bounds(train, cols=["x"])
    out = apply_bounds(pd.DataFrame({"x": [-50.0, 1.0, 50.0, 99.0, 500.0]}), bounds)
    assert out["x"].tolist() == pytest.approx([0.0, 0.0, 49 / 98, 1.0, 1.0])


def test_bounds_come_from_training_data_only():
    train = pd.DataFrame({"x": range(101)}, dtype=float)
    test = pd.DataFrame({"x": [1000.0, 2000.0]})
    bounds = fit_bounds(train, cols=["x"])
    assert apply_bounds(test, bounds)["x"].tolist() == [1.0, 1.0]
