## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.heston         # compare Heston vs BS prices
python -m src.smile          # extract the implied-vol smile
python -m src.make_figure    # regenerate figures (takes 2-3 min)
pytest -v                    # ~9 tests
```

## Verification

The implementation is checked against known properties:
- **Heston reduces to BS when ξ→0**: zero vol-of-vol eliminates the stochastic-vol component, and the MC price matches the BS closed form within Monte Carlo error.
- **Put-call parity holds under Heston**: C − P = S − K·exp(−rT), verified within combined standard errors.
- **Variance mean-reverts**: over many paths with fast reversion (κ=5), the average terminal variance is near θ.
- **Negative ρ produces skew**: OTM puts have higher implied vol than OTM calls when ρ < 0.

## Notes and extensions

This implementation uses Euler discretization with full truncation for the variance process. More sophisticated schemes exist: the Broadie-Kaya exact simulation (no discretization error, but slower), the QE scheme of Andersen (2008), and the Alfonsi scheme. For the price accuracy we're targeting (~0.01 precision on option prices), Euler with full truncation and 100 timesteps is sufficient.

The Heston model also has a semi-analytical solution via characteristic functions (the Heston formula), which gives option prices without Monte Carlo. This is faster and more accurate for European options but doesn't extend to path-dependent payoffs. A natural extension would be implementing the characteristic-function approach and cross-validating against the MC prices shown here.