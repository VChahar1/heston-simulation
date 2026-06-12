from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from .heston import heston_european_call


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call price (used for implied-vol inversion)."""
    if sigma <= 0 or T <= 0:
        return max(S - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def implied_vol_from_price(
    market_price: float, S: float, K: float, T: float, r: float,
    sigma_low: float = 1e-4, sigma_high: float = 3.0,
) -> float:
    """Recover BS implied vol from an option price via Brent's method."""
    def objective(sigma):
        return bs_call_price(S, K, T, r, sigma) - market_price

    f_low = objective(sigma_low)
    f_high = objective(sigma_high)
    if f_low * f_high > 0:
        return float("nan")

    return float(brentq(objective, sigma_low, sigma_high, xtol=1e-8))


def extract_smile(
    S0: float, T: float, r: float,
    v0: float, kappa: float, theta: float, xi: float, rho: float,
    strikes: list[float] | np.ndarray,
    n_paths: int = 200_000,
    n_steps: int = 100,
    seed: int | None = None,
) -> dict:
    """Price calls at each strike under Heston, extract implied vols.

    Returns a dict with 'strikes', 'prices', 'implied_vols', and 'stderrs'.
    """
    prices, stderrs, ivols = [], [], []

    for K in strikes:
        result = heston_european_call(
            S0, K, T, r, v0, kappa, theta, xi, rho,
            n_paths=n_paths, n_steps=n_steps, seed=seed,
        )
        prices.append(result["price"])
        stderrs.append(result["stderr"])

        iv = implied_vol_from_price(result["price"], S0, K, T, r)
        ivols.append(iv)

    return {
        "strikes": np.array(strikes),
        "prices": np.array(prices),
        "stderrs": np.array(stderrs),
        "implied_vols": np.array(ivols),
    }


def _demo() -> None:
    S0, T, r = 100, 1.0, 0.05
    v0, kappa, theta, xi, rho = 0.04, 2.0, 0.04, 0.3, -0.7

    strikes = np.linspace(70, 130, 13)
    result = extract_smile(S0, T, r, v0, kappa, theta, xi, rho,
                           strikes, n_paths=100_000, seed=0)

    print("Heston implied-volatility smile")
    print("=" * 45)
    print(f"{'Strike':>8s}  {'Price':>8s}  {'Impl Vol':>8s}")
    print("-" * 45)
    for K, p, iv in zip(result["strikes"], result["prices"], result["implied_vols"]):
        print(f"{K:>8.1f}  {p:>8.4f}  {iv:>8.4f}")
    print()
    print(f"σ_BS(ATM) = √v0 = {np.sqrt(v0):.4f}")
    print("Notice: OTM puts (low K) have higher implied vol than OTM calls (high K).")
    print("This skew comes from ρ < 0 (the leverage effect).")


if __name__ == "__main__":
    _demo()