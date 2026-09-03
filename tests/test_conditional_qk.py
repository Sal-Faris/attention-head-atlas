import unittest

import numpy as np

from head_atlas.conditional_qk import (
    benjamini_hochberg,
    document_bootstrap_mean,
    fit_conditional_qk_subspace,
    mapped_residual_projector,
    normalized_chordal_distance,
    query_feature_margin,
    trace_shrinkage,
)


class ConditionalQKTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(8)
        self.queries = rng.normal(size=(1000, 4))
        map_matrix = np.asarray(
            [[2.0, 0.0, 0.0, 0.0], [0.0, 0.3, 0.0, 0.0], [0.0, 0.0, 0.1, 0.0], [0.0, 0.0, 0.0, 0.05]]
        )
        self.negative_keys = rng.normal(scale=0.15, size=(1000, 4))
        self.positive_keys = self.negative_keys + self.queries @ map_matrix

    def test_fit_recovers_predictive_conditional_direction(self):
        fit = fit_conditional_qk_subspace(
            self.queries,
            self.positive_keys,
            self.negative_keys,
            rank=1,
            shrinkage=1e-3,
        )
        feature, residual, mean = query_feature_margin(
            self.queries, self.positive_keys, self.negative_keys, fit
        )
        full = np.sum(self.queries * (self.positive_keys - self.negative_keys), axis=1)

        self.assertGreater(fit.singular_values[0], fit.singular_values[1])
        np.testing.assert_allclose(feature + residual + mean, full, atol=1e-8)
        self.assertGreater(np.corrcoef(feature, full)[0, 1] ** 2, 0.9)

    def test_mapped_projectors_are_invariant_to_query_key_coordinate_gauge(self):
        fit = fit_conditional_qk_subspace(
            self.queries,
            self.positive_keys,
            self.negative_keys,
            rank=1,
            shrinkage=0.0,
        )
        rng = np.random.default_rng(9)
        gauge = rng.normal(size=(4, 4))
        while abs(np.linalg.det(gauge)) < 0.1:
            gauge = rng.normal(size=(4, 4))
        inverse_transpose = np.linalg.inv(gauge).T
        transformed_fit = fit_conditional_qk_subspace(
            self.queries @ gauge,
            self.positive_keys @ inverse_transpose,
            self.negative_keys @ inverse_transpose,
            rank=1,
            shrinkage=0.0,
        )
        query_reader = rng.normal(size=(11, 4))
        key_reader = rng.normal(size=(11, 4))
        query_projector = mapped_residual_projector(query_reader, fit.query_basis)
        key_projector = mapped_residual_projector(key_reader, fit.key_basis)
        transformed_query_projector = mapped_residual_projector(
            query_reader @ gauge, transformed_fit.query_basis
        )
        transformed_key_projector = mapped_residual_projector(
            key_reader @ inverse_transpose, transformed_fit.key_basis
        )

        self.assertLess(normalized_chordal_distance(query_projector, transformed_query_projector), 1e-7)
        self.assertLess(normalized_chordal_distance(key_projector, transformed_key_projector), 1e-7)

    def test_shrinkage_bootstrap_and_bh_adjustment(self):
        covariance = np.asarray([[2.0, 0.5], [0.5, 1.0]])
        shrunk = trace_shrinkage(covariance, 1.0)
        np.testing.assert_allclose(shrunk, np.eye(2) * 1.5)
        adjusted = benjamini_hochberg(np.asarray([0.01, 0.04, 0.03]))
        np.testing.assert_allclose(adjusted, [0.03, 0.04, 0.04])
        summary = document_bootstrap_mean(
            np.asarray([1.0, 3.0, 5.0, 7.0]),
            np.asarray([0, 0, 1, 1]),
            repetitions=99,
            rng=np.random.default_rng(10),
        )
        self.assertAlmostEqual(summary["mean"], 4.0)
        self.assertLess(summary["lower_95"], summary["upper_95"])


if __name__ == "__main__":
    unittest.main()
