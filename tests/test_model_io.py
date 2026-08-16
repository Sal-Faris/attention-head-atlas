import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from head_atlas.model_io import (
    extract_from_transformer_lens,
    load_operator_bundle,
    save_operator_bundle,
    verify_extracted_actions,
)
from head_atlas.operators import HeadOperator


class ModelIoTests(unittest.TestCase):
    def test_transformer_lens_adapter_matches_direct_head_computation(self):
        generator = torch.Generator().manual_seed(7)
        fake_model = SimpleNamespace(
            cfg=SimpleNamespace(n_layers=2, n_heads=3),
            W_V=torch.randn(2, 3, 5, 2, generator=generator),
            W_O=torch.randn(2, 3, 2, 5, generator=generator),
            W_Q=torch.randn(2, 3, 5, 2, generator=generator),
            W_K=torch.randn(2, 3, 5, 2, generator=generator),
        )
        states = torch.randn(4, 5, generator=generator)

        ov_operators = extract_from_transformer_lens(fake_model, "OV")
        qk_operators = extract_from_transformer_lens(fake_model, "QK")

        ov = torch.from_numpy(ov_operators[5].matrix)
        qk = torch.from_numpy(qk_operators[5].matrix)
        expected_ov = (states @ fake_model.W_V[1, 2]) @ fake_model.W_O[1, 2]
        expected_qk = (states @ fake_model.W_Q[1, 2]) @ (states @ fake_model.W_K[1, 2]).T
        torch.testing.assert_close(states @ ov, expected_ov)
        torch.testing.assert_close(states @ qk @ states.T, expected_qk)
        ov_error = verify_extracted_actions(fake_model, ov_operators)
        qk_error = verify_extracted_actions(fake_model, qk_operators)
        self.assertLess(ov_error["maximum_relative_error"], 1e-5)
        self.assertLess(qk_error["maximum_relative_error"], 1e-5)

    def test_operator_bundle_round_trip(self):
        operators = [
            HeadOperator(0, 0, "OV", np.eye(3)),
            HeadOperator(0, 1, "OV", np.diag([1.0, 2.0, 3.0])),
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "operators.npz"
            save_operator_bundle(path, operators, {"model": "synthetic"})
            loaded, metadata = load_operator_bundle(path)
        self.assertEqual(metadata["model"], "synthetic")
        self.assertEqual([(x.layer, x.head, x.kind) for x in loaded], [(0, 0, "OV"), (0, 1, "OV")])
        np.testing.assert_array_equal(loaded[1].matrix, operators[1].matrix)


if __name__ == "__main__":
    unittest.main()
