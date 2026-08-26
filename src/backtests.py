"""Statistical backtests for VaR/ES forecasts.

A "breach" on day t is the event that the realized return was worse than
the forecast VaR for that day. Given alpha, a well-calibrated model should
breach at rate alpha, and breaches should not cluster in time (clustering
means the model isn't reacting to changing risk conditions). Separately,
on the days a breach does occur, a well-calibrated ES forecast should
match the *size* of the loss on average — that's what `es_backtest` below
checks, since a model can get the breach rate and clustering right while
still being systematically wrong about how bad a breach is.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

def breach_indicator(actual: np.ndarray, var_forecast: np.ndarray) -> np.ndarray:
    """Compute the boolean breach sequence.

    A breach occurs when the realized return is worse than (more negative
    than) the negative of the forecast VaR, i.e. the loss exceeds VaR.

    Args:
        actual: realized log returns, shape (n,).
        var_forecast: forecast VaR (positive magnitude) for each day,
            shape (n,), aligned with `actual`.

    Returns:
        Boolean array of shape (n,), True where a breach occurred.
    """
    actual = np.asarray(actual, dtype=float)
    var_forecast = np.asarray(var_forecast, dtype=float)
    return actual < -var_forecast

def _log_term(count: int, probability: float) -> float:
    """Return count * log(probability), handling 0 * log(0) as 0."""
    if count == 0:
        return 0.0

    if probability == 0:
        return -np.inf

    return count * np.log(probability)

def kupiec_pof(breaches: np.ndarray, alpha: float) -> tuple[float, float]:
    """Kupiec (1995) proportion-of-failures (unconditional coverage) test.

    Tests H0: the true breach probability equals `alpha`, against a
    binomial alternative, via a likelihood-ratio statistic.

    Let n = len(breaches), x = number of breaches (True values), and
    pi_hat = x / n the observed breach rate. The LR statistic is:

        LR_pof = -2 * log(
            (alpha^x * (1 - alpha)^(n - x))
            / (pi_hat^x * (1 - pi_hat)^(n - x))
        )

    Under H0, LR_pof ~ chi-square with 1 degree of freedom. Handle the
    edge cases x=0 and x=n (pi_hat^x or (1-pi_hat)^(n-x) is 0^0) by
    treating 0^0 = 1 in the log-likelihood (i.e. drop that term).

    Args:
        breaches: boolean array from `breach_indicator`.
        alpha: the nominal tail probability the VaR model targets.

    Returns:
        (LR_statistic, p_value). p_value from
        `scipy.stats.chi2.sf(LR_statistic, df=1)`.
    """
    breaches = np.asarray(breaches, dtype=bool)

    n = len(breaches)
    x = int(breaches.sum())

    pi_hat = x / n

    log_L0 = (_log_term(x, alpha)+ _log_term(n - x, 1 - alpha))

    log_L1 = (_log_term(x, pi_hat)+ _log_term(n - x, 1 - pi_hat))

    LR = -2 * (log_L0 - log_L1)

    p_value = stats.chi2.sf(LR, df=1)

    return float(LR), float(p_value)


def christoffersen_independence(breaches: np.ndarray) -> tuple[float, float]:
    """Christoffersen (1998) test of independence of breaches.

    Tests H0: breaches are independent over time (no clustering), against
    the alternative that breach probability depends on whether yesterday
    was a breach (a first-order Markov chain).

    Build the transition counts from the breach sequence:
        n00 = count of (not breach -> not breach)
        n01 = count of (not breach -> breach)
        n10 = count of (breach -> not breach)
        n11 = count of (breach -> breach)

    Transition probabilities:
        pi01 = n01 / (n00 + n01)
        pi11 = n11 / (n10 + n11)
        pi   = (n01 + n11) / (n00 + n01 + n10 + n11)   # pooled

    LR statistic:
        LR_ind = -2 * log(
            (1-pi)^(n00+n10) * pi^(n01+n11)
        ) + 2 * log(
            (1-pi01)^n00 * pi01^n01 * (1-pi11)^n10 * pi11^n11
        )

    Under H0, LR_ind ~ chi-square with 1 degree of freedom. Handle
    degenerate cases (a transition count that never occurs) by dropping
    the corresponding 0^0 term, same as in `kupiec_pof`.

    Args:
        breaches: boolean array from `breach_indicator`.

    Returns:
        (LR_statistic, p_value).
    """
    breaches = np.asarray(breaches, dtype=bool)

    previous = breaches[:-1]

    current = breaches[1:]

    n00 = int(np.sum((~previous) & (~current)))
    n01 = int(np.sum((~previous) & current))
    n10 = int(np.sum(previous & (~current)))
    n11 = int(np.sum(previous & current))

    denom_0 = n00 + n01
    denom_1 = n10 + n11

    pi01 = n01 / denom_0 if denom_0 > 0 else 0.0
    pi11 = n11 / denom_1 if denom_1 > 0 else 0.0

    total = n00 + n01 + n10 + n11
    pi = (n01 + n11) / total

    log_L0 = (_log_term(n00 + n10, 1 - pi)+ _log_term(n01 + n11, pi))

    log_L1 = (
        _log_term(n00, 1 - pi01)
        + _log_term(n01, pi01)
        + _log_term(n10, 1 - pi11)
        + _log_term(n11, pi11)
    )

    LR = -2 * (log_L0 - log_L1)

    p_value = stats.chi2.sf(LR, df=1)

    return float(LR), float(p_value)


def christoffersen_cc(breaches: np.ndarray, alpha: float) -> tuple[float, float]:
    """Christoffersen conditional coverage (CC) test.

    Joint test of correct unconditional coverage (Kupiec) AND
    independence (Christoffersen), combined as:

        LR_cc = LR_pof + LR_ind

    where LR_pof is from `kupiec_pof` and LR_ind is from
    `christoffersen_independence`. Under H0, LR_cc ~ chi-square with 2
    degrees of freedom (1 from each component test).

    Args:
        breaches: boolean array from `breach_indicator`.
        alpha: the nominal tail probability the VaR model targets.

    Returns:
        (LR_statistic, p_value). p_value from
        `scipy.stats.chi2.sf(LR_statistic, df=2)`.
    """
    LR_pof, _ = kupiec_pof(breaches, alpha)
    LR_ind, _ = christoffersen_independence(breaches)

    LR_cc = LR_pof + LR_ind

    p_value = stats.chi2.sf(LR_cc, df=2)

    return float(LR_cc), float(p_value)


def es_backtest(
    actual: np.ndarray,
    var_forecast: np.ndarray,
    es_forecast: np.ndarray,
    n_bootstrap: int = 10_000,
    random_state: int | None = None,
) -> tuple[float, float]:
    """McNeil & Frey (2000) bootstrap test of Expected Shortfall accuracy.

    Kupiec/Christoffersen only check whether breaches happen at the right
    RATE and are INDEPENDENT — they say nothing about how bad a breach is.
    On each breach day (actual < -var_forecast), define the realized loss
    magnitude and the shortfall residual:

        loss_t = -actual_t
        e_t    = loss_t - es_forecast_t

    By definition, ES is the expected loss *given* a breach, so under a
    correctly calibrated model E[e_t] = 0 across breach days:

        H0: E[e_t] = 0   (ES forecast is accurate on average)
        H1: E[e_t] > 0   (realized losses are systematically worse than
                          the ES forecast predicted)

    One-sided, since underestimating tail severity is the failure mode
    that matters for risk management. Tested via a nonparametric
    percentile bootstrap rather than a parametric test, since the number
    of breach days is small and e_t isn't normally distributed: resample
    {e_t} with replacement `n_bootstrap` times, and the p-value is the
    fraction of bootstrap sample means that fall at or below zero.

    Requires at least 2 breach days; returns (nan, nan) otherwise (same
    convention as the degenerate cases in `kupiec_pof` and
    `christoffersen_independence`).

    Args:
        actual: realized log returns, shape (n,).
        var_forecast: forecast VaR (positive magnitude), shape (n,).
        es_forecast: forecast ES (positive magnitude), shape (n,).
        n_bootstrap: number of bootstrap resamples.
        random_state: seed for reproducibility.

    Returns:
        (mean_residual, p_value). A p_value near 0 means ES is
        systematically underestimating how bad breaches actually are.

    Reference:
        McNeil, A. J., & Frey, R. (2000). "Estimation of Tail-Related Risk
        Measures for Heteroscedastic Financial Time Series: An Extreme
        Value Approach." Journal of Empirical Finance, 7(3-4), 271-300.
    """
    actual = np.asarray(actual, dtype=float)
    var_forecast = np.asarray(var_forecast, dtype=float)
    es_forecast = np.asarray(es_forecast, dtype=float)

    if not (
        actual.shape == var_forecast.shape == es_forecast.shape
    ):
        raise ValueError(
            "actual, var_forecast, and es_forecast must have the same shape"
        )

    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")

    breaches = breach_indicator(actual, var_forecast)

    if breaches.sum() < 2:
        return np.nan, np.nan

    losses = -actual[breaches]

    residuals = losses - es_forecast[breaches]

    mean_residual = float(residuals.mean())

    rng = np.random.default_rng(random_state)

    n_breaches = len(residuals)

    bootstrap_samples = rng.choice(
        residuals,
        size=(n_bootstrap, n_breaches),
        replace=True,
    )

    bootstrap_means = bootstrap_samples.mean(axis=1)

    p_value = float(np.mean(bootstrap_means <= 0))

    return mean_residual, p_value
