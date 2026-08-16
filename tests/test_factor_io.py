import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from safetensors.numpy import save_file

from head_atlas.factor_io import (
    extract_factors_from_transformer_lens,
    extract_processed_factors_from_safetensors,
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

    def test_gpt2_safetensors_extraction_applies_processing(self):
        rng = np.random.default_rng(4)
        d_model, n_heads = 6, 2
        qkv = rng.standard_normal((d_model, 3 * d_model)).astype(np.float32)
        output = rng.standard_normal((d_model, d_model)).astype(np.float32)
        layer_norm = rng.uniform(0.1, 0.9, d_model).astype(np.float32)
        with tempfile.TemporaryDirectory() as temporary_directory:
            snapshot = Path(temporary_directory)
            (snapshot / "config.json").write_text(
                '{"model_type":"gpt2","n_layer":1,"n_head":2,"n_embd":6}',
                encoding="utf-8",
            )
            save_file(
                {
                    "h.0.attn.c_attn.weight": qkv,
                    "h.0.attn.c_proj.weight": output,
                    "h.0.ln_1.weight": layer_norm,
                },
                snapshot / "model.safetensors",
            )
            qk, metadata = extract_processed_factors_from_safetensors(snapshot, "QK")
            ov, _ = extract_processed_factors_from_safetensors(snapshot, "OV")

        width = d_model // n_heads
        expected_q = qkv[:, :width] * layer_norm[:, None]
        expected_q -= expected_q.mean(axis=0, keepdims=True)
        expected_v = qkv[:, 2 * d_model : 2 * d_model + width] * layer_norm[:, None]
        expected_v -= expected_v.mean(axis=0, keepdims=True)
        expected_o = output[:width]
        expected_o -= expected_o.mean(axis=-1, keepdims=True)
        np.testing.assert_allclose(qk[0].left, expected_q)
        np.testing.assert_allclose(ov[0].left, expected_v)
        np.testing.assert_allclose(ov[0].right, expected_o.T)
        self.assertEqual(metadata["model_type"], "gpt2")

    def test_neox_safetensors_extraction_respects_head_qkv_order(self):
        rng = np.random.default_rng(9)
        d_model, n_heads = 8, 2
        qkv = rng.standard_normal((3 * d_model, d_model)).astype(np.float32)
        output = rng.standard_normal((d_model, d_model)).astype(np.float32)
        layer_norm = rng.uniform(0.1, 0.9, d_model).astype(np.float32)
        with tempfile.TemporaryDirectory() as temporary_directory:
            snapshot = Path(temporary_directory)
            (snapshot / "config.json").write_text(
                '{"model_type":"gpt_neox","num_hidden_layers":1,'
                '"num_attention_heads":2,"hidden_size":8,"rotary_pct":0.25}',
                encoding="utf-8",
            )
            save_file(
                {
                    "gpt_neox.layers.0.attention.query_key_value.weight": qkv,
                    "gpt_neox.layers.0.attention.dense.weight": output,
                    "gpt_neox.layers.0.input_layernorm.weight": layer_norm,
                },
                snapshot / "model.safetensors",
            )
            operators, metadata = extract_processed_factors_from_safetensors(snapshot, "QK")

        width = d_model // n_heads
        expected_q = qkv[:width].T * layer_norm[:, None]
        expected_q -= expected_q.mean(axis=0, keepdims=True)
        expected_second_q = qkv[3 * width : 4 * width].T * layer_norm[:, None]
        expected_second_q -= expected_second_q.mean(axis=0, keepdims=True)
        np.testing.assert_allclose(operators[0].left, expected_q)
        np.testing.assert_allclose(operators[1].left, expected_second_q)
        self.assertEqual(metadata["rotary_pct"], 0.25)


if __name__ == "__main__":
    unittest.main()
