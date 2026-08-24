"""Walk-forward VaR/ES backtest runner.

Usage:
    python run.py --ticker SPY --alpha 0.05 --window 250 --start 2015-01-01

For each model in MODELS, rolls a fixed-size window of past returns
forward through the sample, forecasting next-day VaR/ES at each step with
no lookahead, then runs the Kupiec, Christoffersen, and McNeil-Frey ES
backtests on the resulting forecasts and saves a breach plot.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")  # headless: this script only ever saves a PNG

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtests import (
    breach_indicator,
    kupiec_pof,
    christoffersen_independence,
    christoffersen_cc,
    es_backtest,
)
from src.data import load_returns
from src.models import (
    garch_var_es,
    gaussian_var_es,
    historical_var_es,
    student_t_var_es,
)

ModelFn = Callable[..., tuple[float, float]]

MODELS: dict[str, ModelFn] = {
    "historical": historical_var_es,
    "gaussian": gaussian_var_es,
    "student_t": student_t_var_es,
    "garch": garch_var_es,
}

REPORTS_DIR = Path(__file__).resolve().parent / "reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPY", help="ticker symbol")
    parser.add_argument(
        "--alpha", type=float, default=0.05, help="tail probability"
    )
    parser.add_argument(
        "--window", type=int, default=250, help="rolling window size (days)"
    )
    parser.add_argument(
        "--start",
        default="2015-01-01",
        help="start date for the return series (ISO format)",
    )
    parser.add_argument(
        "--end",
        default=dt.date.today().isoformat(),
        help="end date for the return series (ISO format)",
    )
    return parser.parse_args()


def rolling_forecasts(
    returns: pd.Series, model_fn: ModelFn, window: int, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Walk a fixed-size window forward, forecasting VaR/ES at each step.

    For forecast day t, the model sees only:
        returns[t-window:t]

    Therefore return[t] itself is never used in producing its own forecast.

    Returns:
        (actual, var_forecast, es_forecast), each shape (len(returns) -
        window,).
    """
    values = returns.to_numpy(dtype=float)
    n = len(values)

    if n <= window:
        raise ValueError(
            f"Need more than {window} returns, but only {n} are available."
        )

    actual = np.empty(n - window)
    var_forecast = np.empty(n - window)
    es_forecast = np.empty(n - window)

    for i, t in enumerate(range(window, n)):
        # IMPORTANT:
        # Python slicing excludes the right endpoint, so this uses
        # t-window, ..., t-1 and does NOT include return[t].
        train_window = values[t - window : t]

        var_t, es_t = model_fn(train_window, alpha)

        actual[i] = values[t]
        var_forecast[i] = var_t
        es_forecast[i] = es_t

    return actual, var_forecast, es_forecast


def safe_backtests(
    breaches: np.ndarray,
    alpha: float,
) -> tuple[float, float, float, float, float, float]:
    """Run VaR backtests while handling degenerate breach sequences.

    Returns:
        kupiec_stat, kupiec_p,
        ind_stat, ind_p,
        cc_stat, cc_p

    Undefined statistics are returned as np.nan.
    """
    breaches = np.asarray(breaches, dtype=bool)

    n = len(breaches)
    x = int(breaches.sum())

    # Kupiec is still mathematically computable with x=0 or x=n
    # if src/backtests.py handles 0^0 correctly.
    try:
        kupiec_stat, kupiec_p = kupiec_pof(breaches, alpha)
    except (ValueError, ZeroDivisionError, FloatingPointError):
        kupiec_stat, kupiec_p = np.nan, np.nan

    # Independence requires transitions.
    #
    # With fewer than two observations there are none.
    # With all breaches or no breaches, the Markov comparison is
    # degenerate and is not informative, so report NaN instead of
    # pretending that the test passed.
    if n < 2 or x == 0 or x == n:
        ind_stat, ind_p = np.nan, np.nan
        cc_stat, cc_p = np.nan, np.nan
    else:
        try:
            ind_stat, ind_p = christoffersen_independence(breaches)

            if np.isfinite(kupiec_stat) and np.isfinite(ind_stat):
                cc_stat, cc_p = christoffersen_cc(breaches, alpha)
            else:
                cc_stat, cc_p = np.nan, np.nan

        except (ValueError, ZeroDivisionError, FloatingPointError):
            ind_stat, ind_p = np.nan, np.nan
            cc_stat, cc_p = np.nan, np.nan

    return (
        float(kupiec_stat),
        float(kupiec_p),
        float(ind_stat),
        float(ind_p),
        float(cc_stat),
        float(cc_p),
    )


def safe_es_backtest(
    actual: np.ndarray,
    var_forecast: np.ndarray,
    es_forecast: np.ndarray,
    random_state: int = 0,
) -> tuple[float, float]:
    """Run the ES backtest, tolerating too-few-breaches degenerate cases
    the same way `safe_backtests` does for Kupiec/Christoffersen.

    A fixed default `random_state` keeps `run.py`'s output reproducible
    across runs — the underlying test is a bootstrap and would otherwise
    give a slightly different p-value each time.

    Returns:
        (es_stat, es_p), NaN if too few breach days.
    """
    try:
        es_stat, es_p = es_backtest(
            actual, var_forecast, es_forecast, random_state=random_state
        )
    except (
        NotImplementedError,
        ValueError,
        ZeroDivisionError,
        FloatingPointError,
    ):
        es_stat, es_p = np.nan, np.nan

    return float(es_stat), float(es_p)


def verdict_from_pvalues(
    kupiec_p: float,
    ind_p: float,
    es_p: float,
    significance: float = 0.05,
) -> str:
    """Create a human-readable verdict from coverage/independence/severity
    tests: does the model breach at the right rate, are those breaches
    independent over time, and — on the days it does breach — is the
    severity forecast (ES) accurate? NaN inputs (e.g. a backtest that's
    still `NotImplementedError`, or a degenerate breach sequence) are
    reported as N/A rather than silently treated as a pass.
    """
    checks = {"coverage": kupiec_p, "independence": ind_p, "severity": es_p}

    na = [name for name, p in checks.items() if not np.isfinite(p)]
    failed = [
        name
        for name, p in checks.items()
        if np.isfinite(p) and p <= significance
    ]

    if not failed:
        verdict = "PASS" if not na else "PASS (available checks)"
    else:
        verdict = f"FAIL {', '.join(failed)}"

    if na:
        verdict += f"; N/A: {', '.join(na)}"

    return verdict


def plot_var_breaches(
    plot_data: dict[str, dict[str, np.ndarray]],
    alpha: float,
    ticker: str,
    path: Path,
) -> None:
    """Plot realized returns vs. forecast VaR, per model, breaches marked.

    One panel per model, stacked vertically and sharing a date axis: the
    realized return series, the -VaR forecast line beneath it, and a
    marker on every day the realized loss breached that line.

    Args:
        plot_data: name -> {"dates", "actual", "var_forecast", "breaches"},
            all aligned arrays of equal length per model.
        alpha: tail probability, used only for the panel legend labels.
        ticker: used in the figure title.
        path: where to save the PNG.
    """
    names = list(plot_data.keys())
    fig, axes = plt.subplots(
        len(names), 1, figsize=(12, 3 * len(names)), sharex=True
    )
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        d = plot_data[name]
        dates, actual, var_forecast, breaches = (
            d["dates"],
            d["actual"],
            d["var_forecast"],
            d["breaches"],
        )

        ax.axhline(0, color="grey", linewidth=0.5)
        ax.plot(
            dates, actual, color="steelblue", linewidth=0.7,
            label="realized return",
        )
        ax.plot(
            dates, -var_forecast, color="black", linewidth=1.0,
            linestyle="--", label=f"-VaR ({alpha:.0%})",
        )
        ax.scatter(
            dates[breaches], actual[breaches], color="crimson", s=16,
            zorder=5, label=f"breach (n={int(breaches.sum())})",
        )
        ax.set_title(name)
        ax.set_ylabel("log return")
        ax.legend(loc="lower left", fontsize=8)

    axes[-1].set_xlabel("date")
    fig.suptitle(f"{ticker}: VaR forecasts and breaches (alpha={alpha:.0%})")
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if not 0 < args.alpha < 1:
        raise ValueError("--alpha must be strictly between 0 and 1.")

    if args.window < 2:
        raise ValueError("--window must be at least 2.")

    returns = load_returns(args.ticker, args.start, args.end)

    rows = []
    plot_data: dict[str, dict[str, np.ndarray]] = {}

    for name, model_fn in MODELS.items():
        try:
            actual, var_forecast, es_forecast = rolling_forecasts(
                returns,
                model_fn,
                args.window,
                args.alpha,
            )
        except NotImplementedError as exc:
            print(f"[skip] {name}: not implemented yet ({exc})")
            continue

        breaches = breach_indicator(actual, var_forecast)

        plot_data[name] = {
            "dates": returns.index[args.window :],
            "actual": actual,
            "var_forecast": var_forecast,
            "breaches": breaches,
        }

        n_obs = len(breaches)
        expected = args.alpha * n_obs
        actual_breaches = int(breaches.sum())
        breach_rate = actual_breaches / n_obs if n_obs else np.nan

        (
            kupiec_stat,
            kupiec_p,
            ind_stat,
            ind_p,
            cc_stat,
            cc_p,
        ) = safe_backtests(breaches, args.alpha)

        es_stat, es_p = safe_es_backtest(actual, var_forecast, es_forecast)

        verdict = verdict_from_pvalues(kupiec_p, ind_p, es_p)

        rows.append(
            {
                "model": name,
                "n_obs": n_obs,
                "expected_breaches": round(expected, 1),
                "actual_breaches": actual_breaches,
                "breach_rate": breach_rate,
                "kupiec_stat": kupiec_stat,
                "kupiec_p": kupiec_p,
                "ind_stat": ind_stat,
                "christoffersen_p": ind_p,
                "cc_stat": cc_stat,
                "cc_p": cc_p,
                "es_stat": es_stat,
                "es_p": es_p,
                "verdict": verdict,
            }
        )

    if not rows:
        print("No models implemented yet — nothing to report.")
        return

    results = pd.DataFrame(rows)

    # Make terminal output easier to read.
    float_columns = [
        "breach_rate",
        "kupiec_stat",
        "kupiec_p",
        "ind_stat",
        "christoffersen_p",
        "cc_stat",
        "cc_p",
        "es_stat",
        "es_p",
    ]

    results[float_columns] = results[float_columns].round(4)

    print(results.to_string(index=False))

    plot_path = REPORTS_DIR / "var_breaches.png"
    plot_var_breaches(plot_data, args.alpha, args.ticker, plot_path)
    print(f"\nSaved plot to {plot_path}")


if __name__ == "__main__":
    main()