"""
=============================================================================
DATASET GENERATION: Hydrogen Diffusion in Metal Hydrides
=============================================================================
Project: Machine Learning-Based Prediction of Hydrogen Diffusion in Metal Hydrides
Course: Mass Transfer Operations - PBL Laboratory

PURPOSE:
    Generates a physically realistic synthetic dataset using the Arrhenius
    equation and Fick's diffusion framework — NOT random values.

THEORY:
    D(T) = D0 * exp(-Ea / (R * T))
    where:
        D0  = pre-exponential factor (m²/s)
        Ea  = activation energy (J/mol)
        R   = universal gas constant = 8.314 J/(mol·K)
        T   = absolute temperature (K)

    Pressure effect: D_eff = D(T) * (1 + alpha * ln(P/P_ref))
    Particle size effect: D_eff = D(T) / (1 + beta * (dp/dp_ref))

MATERIALS MODELED:
    1. MgH2      - High storage capacity, high activation energy
    2. TiFeH2    - Moderate activation energy, faster kinetics
    3. LaNi5H6   - Low activation energy, best kinetics
=============================================================================
"""

import numpy as np
import pandas as pd
import os

# ─── Physical Constants ───────────────────────────────────────────────────────
R = 8.314          # Universal gas constant [J/(mol·K)]
P_REF = 1.0        # Reference pressure [bar]
DP_REF = 50.0      # Reference particle diameter [µm]

# ─── Material Parameters (from literature) ───────────────────────────────────
# Source: Schlapbach & Züttel, Nature 2001; Grochala & Edwards, Chem. Rev. 2004
MATERIALS = {
    "MgH2": {
        "D0": 1.0e-7,       # Pre-exponential factor [m²/s]
        "Ea": 62000,        # Activation energy [J/mol] (~62 kJ/mol)
        "alpha": 0.08,      # Pressure sensitivity coefficient
        "beta": 0.003,      # Particle size sensitivity coefficient
        "label": 0,
    },
    "TiFeH2": {
        "D0": 5.0e-9,
        "Ea": 35000,        # ~35 kJ/mol
        "alpha": 0.06,
        "beta": 0.002,
        "label": 1,
    },
    "LaNi5H6": {
        "D0": 2.5e-8,
        "Ea": 28000,        # ~28 kJ/mol
        "alpha": 0.05,
        "beta": 0.0015,
        "label": 2,
    },
}


def arrhenius_diffusivity(T: np.ndarray, D0: float, Ea: float) -> np.ndarray:
    """
    Compute diffusion coefficient via the Arrhenius equation.

    D(T) = D0 * exp(-Ea / (R * T))

    Args:
        T   : Temperature array [K]
        D0  : Pre-exponential factor [m²/s]
        Ea  : Activation energy [J/mol]

    Returns:
        D   : Diffusivity array [m²/s]
    """
    return D0 * np.exp(-Ea / (R * T))


def pressure_correction(D: np.ndarray, P: np.ndarray, alpha: float) -> np.ndarray:
    """
    Apply logarithmic pressure correction to diffusivity.
    At elevated H2 pressure more interstitial sites become available,
    modestly increasing effective diffusivity.

    D_p = D * (1 + alpha * ln(P / P_ref))
    """
    return D * (1.0 + alpha * np.log(P / P_REF))


def particle_size_correction(D: np.ndarray, dp: np.ndarray, beta: float) -> np.ndarray:
    """
    Apply inverse particle-size correction.
    Larger particles have longer diffusion paths → lower effective diffusivity.

    D_dp = D / (1 + beta * (dp / dp_ref))
    """
    return D / (1.0 + beta * (dp / DP_REF))


def generate_dataset(
    n_samples_per_material: int = 400,
    noise_fraction: float = 0.03,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate the full synthetic dataset.

    Noise is added as ±3% Gaussian noise to simulate experimental uncertainty —
    consistent with reported measurement errors in diffusion experiments.

    Args:
        n_samples_per_material : int   — rows per material
        noise_fraction         : float — fractional Gaussian noise (σ/D)
        seed                   : int   — random seed for reproducibility

    Returns:
        df : pd.DataFrame with columns:
             Temperature_K, Pressure_bar, Particle_Size_um,
             Material, Material_Label,
             Diffusivity_theoretical, Diffusivity_with_noise
    """
    rng = np.random.default_rng(seed)
    records = []

    for mat_name, params in MATERIALS.items():
        D0    = params["D0"]
        Ea    = params["Ea"]
        alpha = params["alpha"]
        beta  = params["beta"]
        label = params["label"]

        # Operating condition ranges (physically relevant for each hydride)
        T  = rng.uniform(300, 700, n_samples_per_material)      # [K]
        P  = rng.uniform(1.0, 50.0, n_samples_per_material)     # [bar]
        dp = rng.uniform(10.0, 200.0, n_samples_per_material)   # [µm]

        # Step 1 — pure Arrhenius diffusivity
        D_arr = arrhenius_diffusivity(T, D0, Ea)

        # Step 2 — apply pressure correction
        D_p = pressure_correction(D_arr, P, alpha)

        # Step 3 — apply particle-size correction
        D_theo = particle_size_correction(D_p, dp, beta)

        # Step 4 — add experimental-level Gaussian noise
        noise = rng.normal(0, noise_fraction, n_samples_per_material)
        D_noisy = D_theo * (1.0 + noise)

        for i in range(n_samples_per_material):
            records.append(
                {
                    "Temperature_K":           round(T[i], 2),
                    "Pressure_bar":            round(P[i], 3),
                    "Particle_Size_um":        round(dp[i], 2),
                    "Material":                mat_name,
                    "Material_Label":          label,
                    "Diffusivity_theoretical": D_theo[i],
                    "Diffusivity_with_noise":  max(D_noisy[i], 1e-16),  # physical floor
                }
            )

    df = pd.DataFrame(records).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_dataset(n_samples_per_material=400, noise_fraction=0.03)

    out_dir = os.path.dirname(__file__)
    path = os.path.join(out_dir, "hydrogen_diffusion_dataset.csv")
    df.to_csv(path, index=False)

    print("=" * 60)
    print("  Dataset Generation Complete")
    print("=" * 60)
    print(f"  Total samples : {len(df)}")
    print(f"  Saved to      : {path}")
    print("\n  Sample Statistics:")
    print(df.describe().to_string())
    print("\n  Material Distribution:")
    print(df["Material"].value_counts().to_string())
    print("=" * 60)
