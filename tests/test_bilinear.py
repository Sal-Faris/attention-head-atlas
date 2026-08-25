import unittest

import numpy as np

from head_atlas.bilinear import (
    bilinear_scores,
    fit_bilinear_margin_model,
    projected_identity_scores,
    qk_margins,
    r_squared,
)


class BilinearTests(unittest.TestCase):
    def test_low_rank_model_recovers_synthetic_margin_map(self):
        rng = np.random.default_rng(11)
        queries = rng.normal(size=(1000, 6))
        differences = rng.normal(size=(1000, 6))
        queries[:, 2:] = 0.0
        differences[:, 2:] = 0.0
        target = qk_margins(queries, differences)
        model = fit_bilinear_margin_model(
            queries,
            differences,
            rank=2,
            ridge=1e-5,
            iterations=600,
            learning_rate=0.05,
        )

        self.assertGreater(r_squared(bilinear_scores(queries, differences, model), target), 0.98)

    def test_identity_projection_and_r_squared_behave_as_expected(self):
        queries = np.asarray([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0]])
        differences = np.asarray([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
        target = qk_margins(queries, differences)
        prediction = projected_identity_scores(queries, differences, np.eye(2))

        np.testing.assert_allclose(prediction, target)
        self.assertAlmostEqual(r_squared(prediction, target), 1.0)


if __name__ == "__main__":
    unittest.main()
