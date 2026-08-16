import unittest

import numpy as np

from head_atlas.operators import HeadOperator, build_ov, build_qk


class OperatorTests(unittest.TestCase):
    def test_ov_matches_direct_row_vector_computation(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal((7, 5))
        w_v = rng.standard_normal((5, 2))
        w_o = rng.standard_normal((2, 5))
        np.testing.assert_allclose(x @ build_ov(w_v, w_o), (x @ w_v) @ w_o)

    def test_qk_matches_direct_dot_products(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal((6, 5))
        w_q = rng.standard_normal((5, 2))
        w_k = rng.standard_normal((5, 2))
        expected = (x @ w_q) @ (x @ w_k).T
        np.testing.assert_allclose(x @ build_qk(w_q, w_k) @ x.T, expected)

    def test_metadata_rejects_non_square_operator(self):
        with self.assertRaises(ValueError):
            HeadOperator(0, 0, "OV", np.ones((2, 3)))


if __name__ == "__main__":
    unittest.main()
