import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from head_atlas.factor_io import (
    extract_factors_from_transformer_lens,
    load_factor_bundle,
    save_factor_bundle,
    verify_factorized_actions,
)


class FactorIoTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(31)
        self.model = SimpleNamespace(
            cfg=SimpleNamespace(n_layers=2, n_heads=3),
            W_V=torch.randn(2, 3, 5, 2, generator=generator),
            W_O=torch.randn(2, 3, 2, 5, generator=generator),
            W_Q=torch.randn(2, 3, 5, 2, generator=generator),
            W_K=torch.randn(2, 3, 5, 2, generator=generator),
        )

    def test_extraction_matches_direct_computation(self):
        qk = extract_factors_from_transformer_lens(self.model, "QK")
        ov = extract_factors_from_transformer_lens(self.model, "OV")

        self.assertEqual(len(qk), 6)
        self.assertEqual(len(ov), 6)
        np.testing.assert_allclose(
            qk[5].materialize(),
            self.model.W_Q[1, 2].numpy() @ self.model.W_K[1, 2].numpy().T,
        )
        np.testing.assert_allclose(
            ov[5].materialize(),
            self.model.W_V[1, 2].numpy() @ self.model.W_O[1, 2].numpy(),
        )
        self.assertLess(
            verify_factorized_actions(self.model, qk)["maximum_relative_error"],
            1e-6,
        )
        self.assertLess(
            verify_factorized_actions(self.model, ov)["maximum_relative_error"],
            1e-6,
        )

    def test_factor_bundle_round_trip(self):
        operators = extract_factors_from_transformer_lens(self.model, "QK")
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "qk_factors.npz"
            save_factor_bundle(path, operators, {"model": "synthetic"})
            loaded, metadata = load_factor_bundle(path)

        self.assertEqual(metadata["format"], "factorized-head-operator-v1")
        self.assertEqual(metadata["model"], "synthetic")
        self.assertEqual(metadata["operator_count"], 6)
        for expected, actual in zip(operators, loaded, strict=True):
            self.assertEqual(
                (actual.layer, actual.head, actual.kind),
                (expected.layer, expected.head, expected.kind),
            )
            np.testing.assert_array_equal(actual.left, expected.left)
            np.testing.assert_array_equal(actual.right, expected.right)

    def test_bundle_rejects_mixed_kinds(self):
        qk = extract_factors_from_transformer_lens(self.model, "QK")
        ov = extract_factors_from_transformer_lens(self.model, "OV")
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "mixed.npz"
            with self.assertRaisesRegex(ValueError, "one kind"):
                save_factor_bundle(path, [qk[0], ov[0]], {})


if __name__ == "__main__":
    unittest.main()
