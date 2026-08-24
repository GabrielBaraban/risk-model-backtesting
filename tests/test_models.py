"""Tests for src/models.py: exact hand-computed cases, large-sample
convergence to closed-form values, and oracle checks against known
distributions.
"""

import numpy as np
import pytest
from scipy import stats

from src.models import (
    gaussian_var_es,
    garch_var_es,
    historical_var_es,
    student_t_var_es,
)


def test_historical_var_es_exact():
    """Hand-computable case: window = -10..10, alpha=0.05.

    There are 21 points. np.quantile's default linear interpolation puts
    the 5th percentile exactly at -9.0:

        position = 0.05 * (21 - 1) = 1

    so the quantile is the second-smallest observation.

    ES is the mean of all observations <= -9:
        mean(-10, -9) = -9.5.
    """
    window = np.arange(-10.0, 11.0)

    var, es = historical_var_es(window, alpha=0.05)

    assert var == pytest.approx(9.0)
    assert es == pytest.approx(9.5)


def test_gaussian_var_es_large_sample():
    """Large-sample N(0,1) case: VaR/ES approach known values."""
    rng = np.random.default_rng(0)
    window = rng.standard_normal(200_000)

    var, es = gaussian_var_es(window, alpha=0.05)

    assert var == pytest.approx(1.6449, abs=0.05)
    assert es == pytest.approx(2.0627, abs=0.05)


def test_student_t_var_es_matches_theoretical_distribution():
    """Fitted Student-t VaR/ES should approach theoretical values.

    Generate a large sample from a Student-t distribution whose parameters
    are known, fit the model, and compare its forecasts against the
    population VaR and ES.
    """
    rng = np.random.default_rng(12345)

    nu = 5.0
    loc = 0.001
    scale = 0.02
    alpha = 0.05

    window = stats.t.rvs(
        df=nu,
        loc=loc,
        scale=scale,
        size=100_000,
        random_state=rng,
    )

    var, es = student_t_var_es(window, alpha)

    # Standard Student-t alpha quantile.
    z = stats.t.ppf(alpha, df=nu)

    # X = loc + scale * T
    theoretical_q = loc + scale * z
    theoretical_var = -theoretical_q

    # For T ~ t_nu:
    #
    # E[T | T <= z]
    # = -(nu + z^2)/(nu - 1) * f_nu(z)/alpha
    theoretical_tail_mean = -(
        (nu + z**2)
        / (nu - 1)
        * stats.t.pdf(z, df=nu)
        / alpha
    )

    theoretical_es = -(
        loc + scale * theoretical_tail_mean
    )

    assert var == pytest.approx(theoretical_var, abs=0.002)
    assert es == pytest.approx(theoretical_es, abs=0.003)


def test_student_t_es_exceeds_var():
    """Expected Shortfall should be at least as severe as VaR."""
    rng = np.random.default_rng(7)

    window = stats.t.rvs(
        df=4,
        size=20_000,
        random_state=rng,
    )

    var, es = student_t_var_es(window, alpha=0.05)

    assert var > 0
    assert es > var


def test_garch_var_es_reacts_to_volatility_level():
    """A higher-volatility history should produce a larger GARCH VaR.

    Use the exact same zero-mean return pattern at two different scales.
    This isolates volatility: the second series has the same shape and
    mean, but five times the magnitude.
    """
    rng = np.random.default_rng(42)

    base = rng.standard_normal(1000)
    base = base - base.mean()

    low_vol = 0.005 * base
    high_vol = 0.025 * base

    low_var, low_es = garch_var_es(
        low_vol,
        alpha=0.05,
        dist="normal",
    )

    high_var, high_es = garch_var_es(
        high_vol,
        alpha=0.05,
        dist="normal",
    )

    assert low_var > 0
    assert low_es > low_var

    assert high_var > low_var
    assert high_es > low_es


def test_garch_var_es_returns_finite_values():
    """Basic smoke test: GARCH should return usable numeric forecasts."""
    rng = np.random.default_rng(123)

    window = 0.01 * rng.standard_normal(750)

    var, es = garch_var_es(
        window,
        alpha=0.05,
        dist="normal",
    )

    assert np.isfinite(var)
    assert np.isfinite(es)

    assert var > 0
    assert es > var