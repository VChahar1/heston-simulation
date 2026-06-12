

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from src.heston import simulate_heston
from src.smile import extract_smile


def figure_sample_paths() -> None:
    """Show a few Heston paths: stock price and variance."""
    S0, v0, r = 100, 0.04, 0.05
    kappa, theta, xi, rho = 2.0, 0.04, 0.3, -0.7
    T, n_steps = 1.0, 252

    S, v = simulate_heston(S0, v0, r, kappa, theta, xi, rho,
                           T, n_steps, n_paths=5, seed=42)
    t_grid = np.linspace(0, T, n_steps + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i in range(5):
        axes[0].plot(t_grid, S[i], linewidth=0.8, alpha=0.8)
    axes[0].set_title("Stock price paths (Heston)")
    axes[0].set_xlabel("Time (years)")
    axes[0].set_ylabel("S")
    axes[0].axhline(S0, color="black", linestyle="--", linewidth=0.5)
    axes[0].grid(True, alpha=0.3)

    for i in range(5):
        axes[1].plot(t_grid, np.sqrt(np.maximum(v[i], 0)) * 100,
                     linewidth=0.8, alpha=0.8)
    axes[1].set_title("Instantaneous volatility paths (√v)")
    axes[1].set_xlabel("Time (years)")
    axes[1].set_ylabel("Vol (%)")
    axes[1].axhline(np.sqrt(theta) * 100, color="black", linestyle="--",
                    linewidth=0.5, label=f"√θ = {np.sqrt(theta)*100:.1f}%")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Heston model: κ={kappa}, θ={theta}, ξ={xi}, ρ={rho}", fontsize=12)
    fig.tight_layout()
    fig.savefig("figures/sample_paths.png", dpi=120)
    print("Saved figures/sample_paths.png")


def figure_smile_by_rho() -> None:
    """Show how rho controls the skew of the implied-vol smile."""
    S0, T, r = 100, 1.0, 0.05
    v0, kappa, theta, xi = 0.04, 2.0, 0.04, 0.3
    strikes = np.linspace(75, 125, 11)

    fig, ax = plt.subplots(figsize=(10, 6))

    for rho_val in [-0.9, -0.5, 0.0, 0.5]:
        result = extract_smile(S0, T, r, v0, kappa, theta, xi, rho_val,
                               strikes, n_paths=100_000, seed=0)
        valid = ~np.isnan(result["implied_vols"])
        ax.plot(result["strikes"][valid],
                result["implied_vols"][valid] * 100,
                "o-", label=f"ρ = {rho_val}", markersize=4)

    ax.axhline(np.sqrt(v0) * 100, color="gray", linestyle="--",
               alpha=0.5, label=f"√v₀ = {np.sqrt(v0)*100:.0f}%")
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Implied volatility (%)")
    ax.set_title("Heston implied-vol smile: effect of ρ (stock-vol correlation)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/smile_by_rho.png", dpi=120)
    print("Saved figures/smile_by_rho.png")


if __name__ == "__main__":
    figure_sample_paths()
    figure_smile_by_rho()