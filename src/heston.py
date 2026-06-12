"""Heston stochastic volatility model: simulation and European option pricing.

Run `python -m src.heston` for a demonstration.
"""

from __future__ import annotations

import numpy as np


def simulate_heston(
    S0: float,
    v0: float,
    r: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate paths under the Heston model.

    Returns
    -------
    S_paths : ndarray of shape (n_paths, n_steps + 1)
        Stock price paths. S_paths[:, 0] = S0.
    v_paths : ndarray of shape (n_paths, n_steps + 1)
        Variance paths. v_paths[:, 0] = v0.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    S = np.empty((n_paths, n_steps + 1))
    v = np.empty((n_paths, n_steps + 1))
    S[:, 0] = S0
    v[:, 0] = v0

    for t in range(n_steps):
        # Correlated Brownian increments.
        Z1 = rng.standard_normal(n_paths)
        Z2 = rng.standard_normal(n_paths)
        W_v = Z1
        W_S = rho * Z1 + np.sqrt(1 - rho**2) * Z2

        # Full truncation: use max(v_t, 0) in the diffusion terms.
        v_pos = np.maximum(v[:, t], 0.0)
        sqrt_v = np.sqrt(v_pos)

        # Variance update (CIR process).
        v[:, t + 1] = (v[:, t]
                       + kappa * (theta - v_pos) * dt
                       + xi * sqrt_v * np.sqrt(dt) * W_v)

        # Stock price update (log-Euler for positivity).
        S[:, t + 1] = S[:, t] * np.exp(
            (r - 0.5 * v_pos) * dt + sqrt_v * np.sqrt(dt) * W_S
        )

    return S, v


def heston_european_call(
    S0: float, K: float, T: float, r: float,
    v0: float, kappa: float, theta: float, xi: float, rho: float,
    n_paths: int = 100_000,
    n_steps: int = 100,
    seed: int | None = None,
) -> dict:
    """Price a European call under the Heston model via Monte Carlo."""
    S, _ = simulate_heston(S0, v0, r, kappa, theta, xi, rho, T,
                           n_steps, n_paths, seed)
    ST = S[:, -1]
    payoffs = np.exp(-r * T) * np.maximum(ST - K, 0.0)
    price = float(payoffs.mean())
    stderr = float(payoffs.std() / np.sqrt(n_paths))
    return {"price": price, "stderr": stderr, "n_paths": n_paths}


def heston_european_put(
    S0: float, K: float, T: float, r: float,
    v0: float, kappa: float, theta: float, xi: float, rho: float,
    n_paths: int = 100_000,
    n_steps: int = 100,
    seed: int | None = None,
) -> dict:
    """Price a European put under the Heston model via Monte Carlo."""
    S, _ = simulate_heston(S0, v0, r, kappa, theta, xi, rho, T,
                           n_steps, n_paths, seed)
    ST = S[:, -1]
    payoffs = np.exp(-r * T) * np.maximum(K - ST, 0.0)
    price = float(payoffs.mean())
    stderr = float(payoffs.std() / np.sqrt(n_paths))
    return {"price": price, "stderr": stderr, "n_paths": n_paths}


def _demo() -> None:
    """Compare Heston vs Black-Scholes (flat vol) prices."""
    from scipy.stats import norm

    S0, K, T, r = 100, 100, 1.0, 0.05

    # Heston parameters (typical equity calibration).
    v0 = 0.04        # initial variance (equiv to 20% vol)
    kappa = 2.0      # mean-reversion speed
    theta = 0.04     # long-run variance (equiv to 20% vol)
    xi = 0.3         # vol-of-vol
    rho = -0.7       # negative correlation (leverage effect)

    heston_call = heston_european_call(
        S0, K, T, r, v0, kappa, theta, xi, rho,
        n_paths=200_000, n_steps=100, seed=0
    )

    # Black-Scholes with sigma = sqrt(v0) for comparison.
    sigma_bs = np.sqrt(v0)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma_bs**2) * T) / (sigma_bs * np.sqrt(T))
    d2 = d1 - sigma_bs * np.sqrt(T)
    bs_call = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    print("Heston vs Black-Scholes: ATM European call")
    print("=" * 55)
    print(f"  Parameters: S0={S0}, K={K}, T={T}, r={r}")
    print(f"  Heston: v0={v0}, κ={kappa}, θ={theta}, ξ={xi}, ρ={rho}")
    print(f"  BS equivalent σ = √v0 = {sigma_bs:.2%}")
    print()
    print(f"  Heston MC price: {heston_call['price']:.4f} ± {heston_call['stderr']:.4f}")
    print(f"  Black-Scholes:   {bs_call:.4f}")
    print(f"  Difference:      {heston_call['price'] - bs_call:.4f}")
    print()
    print("The difference arises from stochastic vol (ξ>0) and the")
    print("leverage effect (ρ<0), which BS cannot capture.")


if __name__ == "__main__":
    _demo()