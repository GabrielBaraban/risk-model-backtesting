"""VaR / Expected Shortfall forecasting models.

Conventions (see also src/data.py):

- Inputs are 1-day log returns (can be negative or positive).
- ``alpha`` is the tail probability (e.g. 0.05 for a 95% VaR/ES).
- VaR and ES are returned as **positive** loss magnitudes:

    VaR = -quantile_alpha(returns)
    ES  = -E[return | return <= quantile_alpha(returns)]

- Each model takes a ``window`` array of past returns (most recent last)
  and forecasts the *next* day's (VaR, ES) as a ``(float, float)`` tuple.
  A model must use only data in ``window`` — no lookahead.

What each model captures, and doesn't (see README.md for backtest results):

- ``historical_var_es``: non-parametric, makes no distributional
  assumption. Fit directly to the empirical tail, so it tracks the breach
  RATE well, but has no notion of time-varying volatility.
- ``gaussian_var_es``: assumes returns are i.i.d. Normal. Real financial
  returns have fatter tails than the Gaussian, so this systematically
  underestimates severe losses.
- ``student_t_var_es``: assumes a Student-t distribution, which has fatter
  tails than the Gaussian and sizes tail severity more accurately — but is
  still fit to an unconditional window, so it doesn't react to changes in
  volatility regime any better than the Gaussian model does.
- ``garch_var_es``: models conditional (time-varying) volatility rather
  than treating returns as i.i.d., so its VaR band reacts to volatility
  clustering. With Gaussian innovations (the default), it still
  underestimates tail severity once volatility is conditioned on.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.special import gammaln
from arch import arch_model

def historical_var_es(window: np.ndarray, alpha: float) -> tuple[float, float]:
    """Historical simulation VaR/ES: empirical quantile of the window.

    Non-parametric — makes no assumption about the shape of the return
    distribution. Uses the empirical alpha-quantile of ``window`` directly.

    Formula:
        q     = empirical alpha-quantile of window   (e.g. np.quantile)
        VaR   = -q
        ES    = -mean(window[window <= q])

    Args:
        window: 1-D array of past log returns, most recent last.
        alpha: tail probability, e.g. 0.05.

    Returns:
        (VaR, ES) as positive loss magnitudes.
    """
    q = np.quantile(window, alpha)

    var = -q
    es = -window[window <= q].mean()

    return float(var), float(es)


def gaussian_var_es(window: np.ndarray, alpha: float) -> tuple[float, float]:
    """Parametric Gaussian VaR/ES.

    Assumes window ~ i.i.d. Normal(mu, sigma^2), fit by sample mean/std.

    Formula (z = Phi^-1(alpha), the standard normal alpha-quantile;
    phi = standard normal PDF, Phi^-1 = scipy.stats.norm.ppf):

        VaR = -(mu + sigma * Phi^-1(alpha))
        ES  = -(mu - sigma * phi(z) / alpha)

    Args:
        window: 1-D array of past log returns, most recent last.
        alpha: tail probability, e.g. 0.05.

    Returns:
        (VaR, ES) as positive loss magnitudes.
    """
    mu = window.mean()
    sigma = window.std(ddof=1)

    z = stats.norm.ppf(alpha)

    var = -(mu + sigma * z)
    es = -(mu - sigma * stats.norm.pdf(z) / alpha)

    return float(var), float(es)


def student_t_var_es(window: np.ndarray, alpha: float) -> tuple[float, float]:
    """Parametric Student-t VaR/ES.

    Assumes window ~ (scaled, shifted) Student-t, fit via
    ``scipy.stats.t.fit(window)`` (returns dof, loc, scale). Student-t has
    fatter tails than Gaussian for finite degrees of freedom, which
    typically fits financial return tails better.

    Formula (analogous to the Gaussian case, but using the fitted t
    distribution's quantile and tail-conditional expectation instead of
    the normal's):

        q   = t.ppf(alpha, dof, loc=loc, scale=scale)
        VaR = -q
        ES  = -E[X | X <= q]  for X ~ t(dof, loc, scale)
            (closed form exists in terms of the t density at the
            standardized quantile and dof; derive it, or estimate it
            numerically from the fitted distribution)

    Args:
        window: 1-D array of past log returns, most recent last.
        alpha: tail probability, e.g. 0.05.

    Returns:
        (VaR, ES) as positive loss magnitudes.
    """
    nu, loc, scale = stats.t.fit(window)

    z = stats.t.ppf(alpha, df=nu)

    q = loc + scale * z
    var = -q

    t_pdf = stats.t.pdf(z, df=nu)

    tail_factor = (
        (nu + z**2)
        / (nu - 1)
        * t_pdf
        / alpha
    )

    es = -(loc - scale * tail_factor)

    return float(var), float(es)


def garch_var_es(
    window: np.ndarray, alpha: float, dist: str = "normal"
) -> tuple[float, float]:
    """GARCH(1,1) conditional-volatility VaR/ES.

    Unlike the other three models, this does NOT treat `window` as i.i.d.
    Instead it fits a GARCH(1,1) model (via the `arch` package) to capture
    volatility clustering, then forecasts next-period conditional
    volatility and combines it with a chosen innovation distribution to
    get VaR/ES.

    Args:
        window: 1-D array of past log returns, most recent last.
        alpha: tail probability, e.g. 0.05.
        dist: innovation distribution for the GARCH fit, e.g. "normal" or
            "t" (passed through to `arch_model(..., dist=dist)`).

    Returns:
        (VaR, ES) as positive loss magnitudes.
    """
    scale = 100.0

    model = arch_model(
        window * scale,
        mean="Constant",
        vol="Garch",
        p=1,
        q=1,
        dist=dist,
    )

    result = model.fit(disp="off")

    forecast = result.forecast(horizon=1)

    mu_next = float(forecast.mean.iloc[-1, 0]) / scale
    sigma_next = float(np.sqrt(forecast.residual_variance.iloc[-1, 0])) / scale

    dist_lower = dist.lower()

    if dist_lower in {"normal", "gaussian"}:
        z = stats.norm.ppf(alpha)

        var = -(mu_next + sigma_next * z)

        tail_factor = stats.norm.pdf(z) / alpha
        es = -(mu_next - sigma_next * tail_factor)

    elif dist_lower in {"t", "studentst", "students-t", "student-t"}:
        distribution = result.model.distribution

        shape_names = distribution.parameter_names()
        shape_params = np.array(
            [result.params[name] for name in shape_names]
        )

        z = float(distribution.ppf(alpha, shape_params))

        var = -(mu_next + sigma_next * z)

        nu = float(shape_params[0])

        # `distribution.partial_moment` standardizes to unit variance
        # (scale = sqrt((nu-2)/nu)), converts z to the raw t_nu domain,
        # applies the Winkler et al. (1972) closed form via the raw
        # (non-log) gamma function, then rescales back. That raw gamma
        # ratio overflows to inf/nan once nu is large enough (roughly
        # nu > 340) that the fitted Student-t is numerically
        # indistinguishable from Gaussian — which happens in practice on
        # low-kurtosis windows. Reproduce the same transform, but with
        # the gamma ratio computed in log-space so it stays finite for
        # any fitted nu.
        scale = np.sqrt((nu - 2) / nu)
        z_raw = z / scale

        log_c = (
            gammaln(0.5 * (nu + 1))
            - gammaln(0.5 * nu)
            - 0.5 * np.log(nu * np.pi)
        )
        e = 0.5 * (nu + 1)
        raw_moment = (0.5 * (np.exp(log_c) * nu) / (1 - e)) * (
            (1 + (z_raw**2) / nu) ** (1 - e)
        )
        partial_mean = scale * raw_moment

        conditional_tail_mean = partial_mean / alpha

        es = -(
            mu_next
            + sigma_next * conditional_tail_mean
        )

    else:
        raise ValueError(
            f"Unsupported GARCH distribution: {dist!r}. "
            "Use 'normal' or 't'."
        )

    return float(var), float(es)
