# risk-model-backtesting

Rigorous backtesting of one-day Value-at-Risk (VaR) and Expected Shortfall
(ES) models: is a risk number actually trustworthy?

## The question

A risk model spits out "tomorrow's 1-day 95% VaR is 2.3%." How do you know
whether to believe it? Three things have to be true:

1. **Rate** — breaches (actual loss exceeding forecast VaR) should happen at
   roughly the rate implied by alpha (e.g. 5% of days for a 95% VaR).
2. **Independence** — breaches should not cluster. A model that breaches 10
   days in a row during a crash and is silent otherwise has the right
   average rate but is dangerously wrong about *when* risk is high.
3. **Severity** — on the days a breach does happen, the ES forecast should
   be an accurate estimate of how bad the loss actually is. A model can get
   the rate and timing right and still badly underestimate tail severity.

This project forecasts one-day VaR/ES with five models of increasing
sophistication, then formally tests all three properties against realized
returns.

## Method

- **Data**: daily adjusted-close prices → 1-day log returns for a single
  ticker, walk-forward (rolling window, no lookahead).
- **Models**: historical simulation, Gaussian (parametric normal), Student-t
  (fat-tailed parametric), GARCH(1,1) (conditional volatility) — tested with
  both Gaussian and Student-t innovations (`garch`, `garch_t`).
- **Backtests**:
  - Kupiec (1995) proportion-of-failures (POF) test — is the breach rate
    correct?
  - Christoffersen (1998) independence test — are breaches i.i.d., or do
    they cluster?
  - Christoffersen conditional coverage (CC) test — joint test of both.
  - McNeil-Frey (2000) bootstrap ES test — is the forecast severity of a
    breach accurate on average?

## Results

`python run.py --ticker SPY --alpha 0.05 --window 250 --start 2015-01-01`
(2013 rolling forecasts, 2016–2023):

| model | expected breaches | actual | Kupiec p | Christoffersen p | ES p | verdict |
|---|---|---|---|---|---|---|
| historical | 100.7 | 101 | 0.9715 | 0.0000 | 0.0240 | FAIL independence, severity |
| gaussian | 100.7 | 105 | 0.6586 | 0.0000 | 0.0000 | FAIL independence, severity |
| student_t | 100.7 | 120 | 0.0545 | 0.0000 | 0.8027 | FAIL independence |
| garch | 100.7 | 118 | 0.0838 | 0.4197 | 0.0000 | FAIL severity |
| garch_t | 100.7 | 130 | 0.0040 | 0.2083 | 0.0869 | FAIL coverage |

![breaches](reports/var_breaches.png)

## Key finding

No model passes all three tests. Historical, Gaussian, and GARCH all get the
breach *rate* roughly right, but historical and Gaussian breaches cluster
tightly around the March 2020 crash (Christoffersen p ≈ 0) — a
constant-width band fit to unconditional history can't react to a sudden
change in volatility regime.

GARCH(1,1) is built to react to exactly that: by forecasting conditional
volatility, its VaR band widens during the crash and narrows afterward, so
its breaches are independent (p = 0.42). But GARCH here still assumes
*Gaussian* shocks once volatility is accounted for, and that assumption is
wrong: even after conditioning on volatility, realized breach-day losses are
worse than a Gaussian tail predicts, so it fails the severity test
decisively (p ≈ 0). Student-t is the mirror image — its fat-tailed
distribution correctly sizes breach severity (p = 0.80), but because it's
still fit to the unconditional window it doesn't react to the volatility
regime shift any better than Gaussian does, and also fails independence.

The natural next experiment is GARCH with Student-t innovations (`garch_t`)
— combining conditional volatility with fat tails. It does fix both prior
failures: independence holds (p = 0.21) and severity holds (p = 0.087). But
it trades them for a new one. It breaches 130 times against 100.7 expected
(6.46% vs. the 5% target), and Kupiec rejects the breach rate decisively
(p = 0.004) — the only model to fail on *coverage*.

The cause isn't the fatter tail itself — a Student-t quantile at the same
alpha is more extreme than the Gaussian one, which should widen VaR, not
tighten it. It's that GARCH's parameters are refit separately under each
likelihood: `garch_t`'s conditional-volatility path isn't "`garch`'s sigma
plus a fatter tail bolted on," it's a different fitted volatility path from
a different MLE, one that on ~72% of days comes out tighter (mean VaR 1.49%
vs. `garch`'s 1.55%). That tightening in the volatility fit outweighs the
tail-widening effect of the fatter quantile, and the breach rate rises
instead of falling.

So combining the two mechanisms didn't produce the model that passes all
three checks — it produced a third, different failure mode. Of the three
more sophisticated models, each isolates exactly one of the three
properties as its failure: Student-t fails independence, GARCH(normal)
fails severity, GARCH-t fails coverage. Getting all three right
simultaneously, on this ticker/window, didn't happen by combining the
fixes for the other two.

## Limitations

- Single-asset, single-ticker analysis — no portfolio effects.
- No regime-switching or structural-break handling in the GARCH model.
- Model parameters are refit on every rolling window — no parameter
  stability analysis.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
python run.py --ticker SPY --alpha 0.05 --window 250 --start 2015-01-01
pytest
```

## References

- Kupiec, P. (1995). "Techniques for Verifying the Accuracy of Risk
  Measurement Models." *Journal of Derivatives*, 3(2), 73-84.
- Christoffersen, P. F. (1998). "Evaluating Interval Forecasts."
  *International Economic Review*, 39(4), 841-862.
- McNeil, A. J., & Frey, R. (2000). "Estimation of Tail-Related Risk
  Measures for Heteroscedastic Financial Time Series: An Extreme Value
  Approach." *Journal of Empirical Finance*, 7(3-4), 271-300.
- McNeil, A. J., Frey, R., & Embrechts, P. *Quantitative Risk Management:
  Concepts, Techniques and Tools.* Princeton University Press.
