"""
=============================================================================
ML MODEL TRAINING: Hydrogen Diffusion Coefficient Prediction
=============================================================================
Project : Machine Learning-Based Prediction of Hydrogen Diffusion in Metal Hydrides
Course  : Mass Transfer Operations — PBL Laboratory

PIPELINE:
    1. Load physics-generated dataset
    2. Feature engineering
    3. Train / evaluate: Linear Regression, Random Forest, Gradient Boosting
    4. Save best model
    5. Produce all scientific visualisations
=============================================================================
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "data", "hydrogen_diffusion_dataset.csv")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
PLOT_DIR    = os.path.join(BASE_DIR, "plots")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)

# ─── Plotting style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":       150,
    "figure.facecolor": "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
})
MATERIAL_COLORS = {"MgH2": "#E63946", "TiFeH2": "#2A9D8F", "LaNi5H6": "#E9C46A"}
ALGO_COLORS     = {"Linear Regression": "#4361EE", "Random Forest": "#2A9D8F",
                   "Gradient Boosting": "#E63946"}

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 — DATA LOADING & FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[Data] Loaded {len(df)} rows from {path}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create physically motivated derived features:
      • 1/T       — Arrhenius x-axis (1/T vs ln D is linear)
      • ln(D)     — target in log-space (spans many orders of magnitude)
      • T²        — non-linear temperature effect
      • ln(P)     — logarithmic pressure dependence
      • T·ln(P)   — interaction term
    """
    df = df.copy()
    df["Inv_Temperature"]   = 1.0 / df["Temperature_K"]
    df["Ln_Diffusivity"]    = np.log(df["Diffusivity_with_noise"])
    df["Temp_Squared"]      = df["Temperature_K"] ** 2
    df["Ln_Pressure"]       = np.log(df["Pressure_bar"])
    df["Temp_Ln_Pressure"]  = df["Temperature_K"] * np.log(df["Pressure_bar"])
    return df


FEATURE_COLS = [
    "Temperature_K",
    "Pressure_bar",
    "Particle_Size_um",
    "Material_Label",
    "Inv_Temperature",
    "Temp_Squared",
    "Ln_Pressure",
    "Temp_Ln_Pressure",
]
TARGET_COL = "Ln_Diffusivity"   # predict in log-space; exponentiate to recover D


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 — MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def build_models() -> dict:
    return {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LinearRegression()),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  RandomForestRegressor(
                n_estimators=200, max_depth=12, min_samples_leaf=2,
                n_jobs=-1, random_state=42
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.08, max_depth=5,
                subsample=0.8, random_state=42
            )),
        ]),
    }


def evaluate(y_true_log, y_pred_log) -> dict:
    """Return R², RMSE, MAE in original (exponential) space."""
    y_true = np.exp(y_true_log)
    y_pred = np.exp(y_pred_log)
    return {
        "R2":   r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE":  mean_absolute_error(y_true, y_pred),
    }


def train_and_evaluate(df: pd.DataFrame):
    df = engineer_features(df)

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.2, random_state=42
    )

    models   = build_models()
    results  = {}
    best_r2  = -np.inf
    best_name = None

    print("\n" + "=" * 60)
    print("  MODEL TRAINING & EVALUATION")
    print("=" * 60)

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred_test = pipe.predict(X_test)

        metrics = evaluate(y_test, y_pred_test)
        cv_r2   = cross_val_score(pipe, X_train, y_train, cv=5,
                                  scoring="r2").mean()

        results[name] = {
            "pipe":      pipe,
            "y_test":    y_test,
            "y_pred":    y_pred_test,
            "metrics":   metrics,
            "cv_r2":     cv_r2,
            "test_idx":  idx_test,
        }

        print(f"\n  [{name}]")
        print(f"    R²   (test)    : {metrics['R2']:.4f}")
        print(f"    RMSE (test)    : {metrics['RMSE']:.4e} m²/s")
        print(f"    MAE  (test)    : {metrics['MAE']:.4e} m²/s")
        print(f"    R²   (CV mean) : {cv_r2:.4f}")

        if metrics["R2"] > best_r2:
            best_r2  = metrics["R2"]
            best_name = name

    print(f"\n  ✓ Best model: {best_name}  (R² = {best_r2:.4f})")

    # Save best model
    joblib.dump(models[best_name], os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(models,            os.path.join(MODEL_DIR, "all_models.pkl"))
    print(f"  ✓ Models saved to {MODEL_DIR}")

    return df, models, results, best_name, X_test, y_test, idx_test


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3 — VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_temperature_vs_diffusivity(df: pd.DataFrame):
    """Arrhenius plot: Temperature vs Diffusivity per material."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Temperature vs Diffusivity (Arrhenius Behaviour)",
                 fontsize=14, fontweight="bold", y=1.02)

    for ax, (mat, color) in zip(axes, MATERIAL_COLORS.items()):
        sub = df[df["Material"] == mat].sort_values("Temperature_K")
        ax.semilogy(sub["Temperature_K"], sub["Diffusivity_theoretical"],
                    color=color, lw=2, label="Theoretical")
        ax.semilogy(sub["Temperature_K"], sub["Diffusivity_with_noise"],
                    "o", color=color, alpha=0.25, ms=3, label="Simulated Data")
        ax.set_xlabel("Temperature (K)", fontsize=11)
        ax.set_ylabel("Diffusivity (m²/s)", fontsize=11)
        ax.set_title(mat, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "01_temperature_vs_diffusivity.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Saved: {path}")


def plot_arrhenius(df: pd.DataFrame):
    """1/T vs ln(D) — linear Arrhenius relationship."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for mat, color in MATERIAL_COLORS.items():
        sub = df[df["Material"] == mat]
        x = 1000 / sub["Temperature_K"]        # 1000/T for readability
        y = np.log(sub["Diffusivity_theoretical"])
        ax.scatter(x, y, color=color, alpha=0.4, s=10, label=mat)

        # Fit line
        coeffs = np.polyfit(x, y, 1)
        x_fit  = np.linspace(x.min(), x.max(), 200)
        ax.plot(x_fit, np.polyval(coeffs, x_fit), color=color, lw=2)

    ax.set_xlabel("1000/T  (K⁻¹)", fontsize=12)
    ax.set_ylabel("ln(D)  [D in m²/s]", fontsize=12)
    ax.set_title("Arrhenius Plot: ln(D) vs 1000/T", fontsize=13, fontweight="bold")
    ax.legend()

    path = os.path.join(PLOT_DIR, "02_arrhenius_plot.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Saved: {path}")


def plot_pressure_vs_diffusivity(df: pd.DataFrame):
    """Pressure vs Diffusivity scatter, one panel per material."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Pressure vs Diffusivity",
                 fontsize=14, fontweight="bold")

    for ax, (mat, color) in zip(axes, MATERIAL_COLORS.items()):
        sub = df[df["Material"] == mat]
        ax.scatter(sub["Pressure_bar"], sub["Diffusivity_theoretical"],
                   color=color, alpha=0.4, s=10, label="Theoretical")
        ax.set_xlabel("Pressure (bar)", fontsize=11)
        ax.set_ylabel("Diffusivity (m²/s)", fontsize=11)
        ax.set_title(mat, fontsize=12, fontweight="bold")
        ax.set_yscale("log")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "03_pressure_vs_diffusivity.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Saved: {path}")


def plot_actual_vs_predicted(results: dict, df: pd.DataFrame):
    """Parity plot (Actual vs Predicted) for all three models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Actual vs Predicted Diffusivity",
                 fontsize=14, fontweight="bold")

    df_eng = engineer_features(df)

    for ax, (name, res) in zip(axes, results.items()):
        y_actual = np.exp(res["y_test"])
        y_pred   = np.exp(res["y_pred"])
        color    = ALGO_COLORS[name]
        metrics  = res["metrics"]

        # Colour by material
        mat_labels = df_eng.loc[res["test_idx"], "Material"].values
        for mat, mc in MATERIAL_COLORS.items():
            mask = mat_labels == mat
            ax.scatter(y_actual[mask], y_pred[mask], color=mc, alpha=0.5,
                       s=10, label=mat)

        # Perfect-prediction line
        lims = [min(y_actual.min(), y_pred.min()),
                max(y_actual.max(), y_pred.max())]
        ax.plot(lims, lims, "k--", lw=1.5, label="Perfect")

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Actual D (m²/s)", fontsize=10)
        ax.set_ylabel("Predicted D (m²/s)", fontsize=10)
        ax.set_title(f"{name}\nR²={metrics['R2']:.4f}", fontsize=11,
                     fontweight="bold")
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "04_actual_vs_predicted.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Saved: {path}")


def plot_feature_importance(results: dict):
    """Feature importance from tree-based models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    tree_models = {k: v for k, v in results.items() if k != "Linear Regression"}

    feature_labels = [
        "Temperature", "Pressure", "Particle Size", "Material",
        "1/T", "T²", "ln(P)", "T·ln(P)",
    ]

    for ax, (name, res) in zip(axes, tree_models.items()):
        estimator = res["pipe"].named_steps["model"]
        importance = estimator.feature_importances_
        idx = np.argsort(importance)
        colors_bar = plt.cm.RdYlGn(np.linspace(0.25, 0.85, len(idx)))

        ax.barh([feature_labels[i] for i in idx], importance[idx],
                color=colors_bar)
        ax.set_xlabel("Importance Score", fontsize=11)
        ax.set_title(f"{name}\nFeature Importance", fontsize=12, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "05_feature_importance.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Saved: {path}")


def plot_correlation_matrix(df: pd.DataFrame):
    """Correlation heatmap of all numeric features."""
    df_eng = engineer_features(df)
    cols = [
        "Temperature_K", "Pressure_bar", "Particle_Size_um",
        "Material_Label", "Inv_Temperature", "Temp_Squared",
        "Ln_Pressure", "Ln_Diffusivity",
    ]
    corr = df_eng[cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Matrix — Features & Diffusivity",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "06_correlation_matrix.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Saved: {path}")


def plot_model_comparison(results: dict):
    """Bar chart comparing R², RMSE, MAE across algorithms."""
    names   = list(results.keys())
    r2s     = [results[n]["metrics"]["R2"]   for n in names]
    rmses   = [results[n]["metrics"]["RMSE"] for n in names]
    maes    = [results[n]["metrics"]["MAE"]  for n in names]
    cv_r2s  = [results[n]["cv_r2"]           for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Model Performance Comparison", fontsize=14, fontweight="bold")

    colors = [ALGO_COLORS[n] for n in names]

    # R²
    axes[0].bar(names, r2s, color=colors, edgecolor="black", linewidth=0.6)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("R²")
    axes[0].set_title("R² Score")
    for i, v in enumerate(r2s):
        axes[0].text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=9)

    # RMSE
    axes[1].bar(names, rmses, color=colors, edgecolor="black", linewidth=0.6)
    axes[1].set_ylabel("RMSE (m²/s)")
    axes[1].set_title("RMSE")
    axes[1].set_yscale("log")

    # MAE
    axes[2].bar(names, maes, color=colors, edgecolor="black", linewidth=0.6)
    axes[2].set_ylabel("MAE (m²/s)")
    axes[2].set_title("MAE")
    axes[2].set_yscale("log")

    plt.xticks(rotation=10)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "07_model_comparison.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # 1. Generate dataset if not present
    if not os.path.exists(DATA_PATH):
        print("[Data] Dataset not found — generating …")
        sys.path.insert(0, os.path.join(BASE_DIR, "data"))
        from generate_dataset import generate_dataset
        df_raw = generate_dataset()
        df_raw.to_csv(DATA_PATH, index=False)
    else:
        df_raw = load_data(DATA_PATH)

    # 2. Train & evaluate
    df, models, results, best_name, X_test, y_test, idx_test = \
        train_and_evaluate(df_raw)

    # 3. Visualisations
    print("\n[Plots] Generating visualisations …")
    plot_temperature_vs_diffusivity(df_raw)
    plot_arrhenius(df_raw)
    plot_pressure_vs_diffusivity(df_raw)
    plot_actual_vs_predicted(results, df_raw)
    plot_feature_importance(results)
    plot_correlation_matrix(df_raw)
    plot_model_comparison(results)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print(f"  Best model : {best_name}")
    print(f"  Plots saved: {PLOT_DIR}")
    print(f"  Models saved: {MODEL_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
