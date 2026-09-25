"""Regression test for the in-place v12 adapter sanitizer."""

from __future__ import annotations

import unittest

import torch

from mjlab_microban.scripts.sanitize_teleop_v12_adapter_checkpoint import (
    _zero_extra_columns_in_place,
)
from mjlab_microban.tasks.microban_teleop_v12_actor import (
    TELEOP_V12_EXTRA_OBSERVATION_COLUMNS,
    TELEOP_V12_SHARED_OBSERVATION_COLUMNS,
)


class TeleopV12SanitizerTest(unittest.TestCase):
    def test_reset_mutates_original_tensor_and_only_extra_columns(self) -> None:
        tensor = torch.ones(512, 83)
        storage = tensor.data_ptr()
        _zero_extra_columns_in_place(tensor)
        self.assertEqual(tensor.data_ptr(), storage)
        self.assertTrue(
            torch.equal(
                tensor[:, TELEOP_V12_EXTRA_OBSERVATION_COLUMNS],
                torch.zeros(512, 20),
            )
        )
        self.assertTrue(
            torch.equal(
                tensor[:, TELEOP_V12_SHARED_OBSERVATION_COLUMNS],
                torch.ones(512, 63),
            )
        )

    def test_reset_rejects_wrong_shape(self) -> None:
        with self.assertRaises(ValueError):
            _zero_extra_columns_in_place(torch.ones(512, 82))


if __name__ == "__main__":
    unittest.main()
