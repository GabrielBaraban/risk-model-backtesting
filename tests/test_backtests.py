"""Tests for src/backtests.py."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from src.backtests import (
    breach_indicator,
    kupiec_pof,
    christoffersen_independence,
    christoffersen_cc,
    es_backtest,
)


def test_breach_indicator():
    actual = np.array([
        -0.01,   # loss smaller than VaR -> no breach
        -0.03,   # loss larger than VaR -> breach
        0.01,    # positive return -> no breach
        -0.02,   # exactly at VaR threshold -> no breach
    ])

    var_forecast = np.array([
        0.02,
        0.02,
        0.02,
        0.02,
    ])

    expected = np.array([
        False,
        True,
        False,
        False,
    ])

    result = breach_indicator(actual, var_forecast)

    np.testing.assert_array_equal(result, expected)


def test_kupiec_expected_breach_rate_has_high_p_value():
    alpha = 0.05
    n = 1000

    breaches = np.zeros(n, dtype=bool)

    # Exactly 50 breaches = 5%
    breaches[::20] = True

    lr, p_value = kupiec_pof(breaches, alpha)

    # Exact observed rate equals nominal alpha,
    # so the unrestricted and null likelihoods coincide.
    assert lr == pytest.approx(0.0, abs=1e-12)
    assert p_value == pytest.approx(1.0, abs=1e-12)


def test_kupiec_too_many_breaches_rejects():
    alpha = 0.05
    n = 1000

    breaches = np.zeros(n, dtype=bool)

    # 150 breaches = 15%, i.e. 3 times the expected rate
    breaches[:150] = True

    lr, p_value = kupiec_pof(breaches, alpha)

    assert lr > 0
    assert p_value < 0.05


def test_kupiec_matches_hand_computed_likelihood_ratio():
    alpha = 0.10

    breaches = np.array([
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ])

    n = len(breaches)
    x = int(breaches.sum())
    pi_hat = x / n

    # Here x/n == alpha exactly.
    expected_lr = -2 * (
        x * np.log(alpha)
        + (n - x) * np.log(1 - alpha)
        - x * np.log(pi_hat)
        - (n - x) * np.log(1 - pi_hat)
    )

    lr, p_value = kupiec_pof(breaches, alpha)

    assert lr == pytest.approx(expected_lr)
    assert p_value == pytest.approx(
        stats.chi2.sf(expected_lr, df=1)
    )


def test_kupiec_zero_breaches_is_finite():
    alpha = 0.05
    breaches = np.zeros(100, dtype=bool)

    lr, p_value = kupiec_pof(breaches, alpha)

    assert np.isfinite(lr)
    assert np.isfinite(p_value)
    assert lr > 0


def test_christoffersen_iid_sequence_does_not_reject():
    # Instead of relying on a tiny random sample, use a large seeded
    # Bernoulli sequence so the test is much less likely to be flaky.
    rng = np.random.default_rng(42)

    alpha = 0.05
    breaches = rng.random(10_000) < alpha

    lr, p_value = christoffersen_independence(breaches)

    assert lr >= 0
    assert p_value > 0.05


def test_christoffersen_clustered_sequence_rejects():
    # Mostly no breaches, followed by one concentrated block.
    breaches = np.zeros(1000, dtype=bool)
    breaches[450:500] = True

    lr, p_value = christoffersen_independence(breaches)

    assert lr > 0
    assert p_value < 0.05


def test_christoffersen_cc_stat_is_sum_of_components():
    breaches = np.array(
        [
            False, False, False, True,
            False, False, True, True,
            False, False, False, False,
            True, False, False, False,
        ],
        dtype=bool,
    )

    alpha = 0.10

    lr_pof, _ = kupiec_pof(breaches, alpha)
    lr_ind, _ = christoffersen_independence(breaches)
    lr_cc, p_cc = christoffersen_cc(breaches, alpha)

    expected_lr = lr_pof + lr_ind
    expected_p = stats.chi2.sf(expected_lr, df=2)

    assert lr_cc == pytest.approx(expected_lr)
    assert p_cc == pytest.approx(expected_p)


def test_christoffersen_cc_p_value_between_zero_and_one():
    breaches = np.array(
        [
            False, False, True, False,
            False, True, False, False,
            False, True, False, False,
        ],
        dtype=bool,
    )

    lr, p_value = christoffersen_cc(breaches, alpha=0.10)

    assert lr >= 0
    assert 0 <= p_value <= 1


def test_es_backtest_well_calibrated_does_not_reject():
    """If breach losses match forecast ES on average, do not reject.

    Breach-day losses are:
        3, 5, 4, 4

    Forecast ES is always 4, so residuals are:
        -1, +1, 0, 0

    Their mean is exactly zero.
    """
    actual = np.array([
        -0.01,
        -0.03,  # breach
        -0.01,
        -0.05,  # breach
        -0.04,  # breach
        -0.01,
        -0.04,  # breach
        -0.01,
    ])

    var_forecast = np.full(len(actual), 0.02)

    es_forecast = np.array([
        0.04,
        0.04,
        0.04,
        0.04,
        0.04,
        0.04,
        0.04,
        0.04,
    ])

    mean_residual, p_value = es_backtest(
        actual,
        var_forecast,
        es_forecast,
        n_bootstrap=10_000,
        random_state=42,
    )

    assert mean_residual == pytest.approx(0.0)
    assert p_value > 0.05


def test_es_backtest_underestimated_es_rejects():
    """If ES forecasts are much too small, the test should reject.

    All breach losses are larger than the ES forecast, so every
    shortfall residual is positive.
    """
    actual = np.array([
        -0.01,
        -0.04,  # breach
        -0.05,  # breach
        -0.01,
        -0.06,  # breach
        -0.04,  # breach
        -0.01,
        -0.05,  # breach
    ])

    var_forecast = np.full(len(actual), 0.02)

    # Intentionally much too low.
    es_forecast = np.full(len(actual), 0.02)

    mean_residual, p_value = es_backtest(
        actual,
        var_forecast,
        es_forecast,
        n_bootstrap=10_000,
        random_state=42,
    )

    assert mean_residual > 0
    assert p_value < 0.05


def test_es_backtest_reproducible_with_fixed_seed():
    """Same inputs and seed should give identical bootstrap results."""
    actual = np.array([
        -0.01,
        -0.03,
        -0.05,
        -0.01,
        -0.04,
        -0.06,
        -0.01,
        -0.035,
    ])

    var_forecast = np.full(len(actual), 0.02)
    es_forecast = np.full(len(actual), 0.04)

    result_1 = es_backtest(
        actual,
        var_forecast,
        es_forecast,
        n_bootstrap=5000,
        random_state=123,
    )

    result_2 = es_backtest(
        actual,
        var_forecast,
        es_forecast,
        n_bootstrap=5000,
        random_state=123,
    )

    assert result_1[0] == pytest.approx(result_2[0])
    assert result_1[1] == pytest.approx(result_2[1])


@pytest.mark.parametrize(
    "actual",
    [
        # No breaches
        np.array([-0.01, -0.01, 0.01, -0.005]),

        # Exactly one breach
        np.array([-0.01, -0.03, 0.01, -0.005]),
    ],
)
def test_es_backtest_too_few_breaches_returns_nan(actual):
    """Zero or one breach should return (nan, nan)."""
    var_forecast = np.full(len(actual), 0.02)
    es_forecast = np.full(len(actual), 0.03)

    mean_residual, p_value = es_backtest(
        actual,
        var_forecast,
        es_forecast,
        random_state=42,
    )

    assert np.isnan(mean_residual)
    assert np.isnan(p_value)