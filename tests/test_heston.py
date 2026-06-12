"""Tests for the Heston simulator and pricer."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from src.heston import simulate_heston, heston_european_call, heston_european_put
from src.smile import implied_vol_from_price, bs_call_price


# ---- Simulation sanity checks ----

def test_paths_start_at_initial_values():
    S, v = simulate_heston(100, 0.04, 0.05, 2.0, 0.04, 0.3, -0.7,
                           1.0, 50, 10, seed=0)
    assert np.all(S[:, 0] == 100)
    assert np.all(v[:, 0] == 0.04)


def test_stock_prices_positive():
    """Log-Euler guarantees S > 0."""
    S, _ = simulate_heston(100, 0.04, 0.05, 2.0, 0.04, 0.5, -0.9,
                           1.0, 100, 10_000, seed=1)
    assert np.all(S > 0)


def test_variance_mean_reverts():
    _, v = simulate_heston(100, 0.01, 0.05, 5.0, 0.04, 0.1, -0.5,
                           5.0, 500, 50_000, seed=2)
    # v0=0.01, theta=0.04, kappa=5 (fast reversion), T=5 years.
    # Terminal variance should be near theta=0.04.
    mean_terminal_v = v[:, -1].mean()
    assert abs(mean_terminal_v - 0.04) < 0.005


def test_feller_condition_comment():

    kappa, theta, xi = 2.0, 0.04, 0.3
    feller = 2 * kappa * theta - xi**2
    # 2*2*0.04 - 0.09 = 0.16 - 0.09 = 0.07 > 0: Feller holds.
    assert feller > 0


# ---- Pricing checks ----

def test_heston_reduces_to_bs_when_xi_zero():
    """When vol-of-vol is zero, Heston = Black-Scholes with sigma=sqrt(v0)."""
    S0, K, T, r, v0 = 100, 100, 1.0, 0.05, 0.04
    sigma_bs = np.sqrt(v0)

    # BS reference.
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma_bs**2) * T) / (sigma_bs * np.sqrt(T))
    d2 = d1 - sigma_bs * np.sqrt(T)
    bs_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    # Heston with xi=0 (no stochastic vol).
    heston = heston_european_call(
        S0, K, T, r, v0, kappa=2.0, theta=v0, xi=0.001, rho=0.0,
        n_paths=200_000, n_steps=100, seed=0,
    )

    assert abs(heston["price"] - bs_price) < 4 * heston["stderr"]


def test_put_call_parity_under_heston():
    """Put-call parity holds for any model: C - P = S - K*exp(-rT)."""
    S0, K, T, r = 100, 105, 1.0, 0.05
    v0, kappa, theta, xi, rho = 0.04, 2.0, 0.04, 0.3, -0.7

    call = heston_european_call(S0, K, T, r, v0, kappa, theta, xi, rho,
                                n_paths=200_000, n_steps=100, seed=0)
    put = heston_european_put(S0, K, T, r, v0, kappa, theta, xi, rho,
                              n_paths=200_000, n_steps=100, seed=0)

    parity_lhs = call["price"] - put["price"]
    parity_rhs = S0 - K * np.exp(-r * T)
    combined_se = np.sqrt(call["stderr"]**2 + put["stderr"]**2)
    assert abs(parity_lhs - parity_rhs) < 4 * combined_se


def test_negative_rho_produces_skew():
    """With rho < 0, OTM puts should have higher implied vol than OTM calls."""
    S0, T, r = 100, 1.0, 0.05
    v0, kappa, theta, xi, rho = 0.04, 2.0, 0.04, 0.3, -0.7

    low_K = 85    # OTM put strike
    high_K = 115  # OTM call strike

    price_low = heston_european_call(S0, low_K, T, r, v0, kappa, theta, xi, rho,
                                     n_paths=200_000, n_steps=100, seed=0)
    price_high = heston_european_call(S0, high_K, T, r, v0, kappa, theta, xi, rho,
                                      n_paths=200_000, n_steps=100, seed=0)

    iv_low = implied_vol_from_price(price_low["price"], S0, low_K, T, r)
    iv_high = implied_vol_from_price(price_high["price"], S0, high_K, T, r)

    assert iv_low > iv_high, (
        f"Expected skew: IV at K={low_K} ({iv_low:.4f}) should exceed "
        f"IV at K={high_K} ({iv_high:.4f}) when ρ<0"
    )


# ---- Implied vol inversion ----

def test_implied_vol_roundtrip():
    """BS price -> implied vol -> BS price should roundtrip."""
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.25
    price = bs_call_price(S, K, T, r, sigma)
    recovered = implied_vol_from_price(price, S, K, T, r)
    assert abs(recovered - sigma) < 1e-6