"""The ONNX parity bound scales with each sample's max |expected|."""

from __future__ import annotations

import numpy as np

from mjlab_microban.scripts.teleop_v12_onnx_gate import (
    ONNX_PARITY_RELATIVE_TOLERANCE,
    parity_bound_ratio,
)


def test_large_outputs_tolerate_float32_rounding_only() -> None:
    expected = np.array([[54.73857, -3.0, 0.0]], dtype=np.float32)
    # A measured export: 2.86e-5 absolute at |out|~55, on a small output of a
    # large-output sample.
    actual = expected + np.array([[0.0, 2.861023e-05, 0.0]], dtype=np.float32)
    assert parity_bound_ratio(actual, expected, atol=2.0e-5) < 1.0
    # The same error on a small output still fails the absolute floor.
    small = np.array([[1.0]], dtype=np.float32)
    assert parity_bound_ratio(small + 2.9e-5, small, atol=2.0e-5) > 1.0
    # A real export defect (O(1e-3) and above) fails at any magnitude.
    assert parity_bound_ratio(expected + 1.0e-3, expected, atol=2.0e-5) > 1.0
    assert ONNX_PARITY_RELATIVE_TOLERANCE == 1.0e-6
