"""
=============================================================================
STREAMLIT APPLICATION: Hydrogen Diffusion Predictor
=============================================================================
Project : Machine Learning-Based Prediction of Hydrogen Diffusion in Metal Hydrides
Course  : Mass Transfer Operations — PBL Laboratory

RUN:
    streamlit run app.py

FEATURES:
    • Sidebar inputs: Temperature, Pressure, Particle Size, Material
    • Predicted diffusivity  (ML model — best saved model)
    • Theoretical diffusivity (Arrhenius + corrections)
    • Percentage error
    • Live Arrhenius curve for the selected material
    • Model performance metrics table
=============================================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
DATA_PATH  = os.path.join(BASE_DIR, "data", "hydrogen_diffusion_dataset.csv")
sys.path.insert(0, os.path.join(BASE_DIR, "data"))

# ─── Physics constants & material parameters ─────────────────────────────────
R = 8.314
P_REF  = 1.0
DP_REF = 50.0

MATERIAL_PARAMS = {
    "MgH₂":      {"D0": 1.0e-7,  "Ea": 62000, "alpha": 0.08,  "beta": 0.003,  "label": 0},
    "TiFeH₂":    {"D0": 5.0e-9,  "Ea": 35000, "alpha": 0.06,  "beta": 0.002,  "label": 1},
    "LaNi₅H₆":   {"D0": 2.5e-8,  "Ea": 28000, "alpha": 0.05,  "beta": 0.0015, "label": 2},
}

MAT_COLORS = {"MgH₂": "#E63946", "TiFeH₂": "#2A9D8F", "LaNi₅H₆": "#E9C46A"}

# Internal keys used in model training (plain ASCII)
MAT_DISPLAY_TO_PLAIN = {
    "MgH₂":    "MgH2",
    "TiFeH₂":  "TiFeH2",
    "LaNi₅H₆": "LaNi5H6",
}


# ─────────────────────────────────────────────────────────────────────────────
#  Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def theoretical_diffusivity(T: float, P: float, dp: float, mat: str) -> float:
    """Compute theoretical D using Arrhenius + pressure + particle-size corrections."""
    p = MATERIAL_PARAMS[mat]
    D0, Ea, alpha, beta = p["D0"], p["Ea"], p["alpha"], p["beta"]
    D = D0 * np.exp(-Ea / (R * T))
    D = D * (1.0 + alpha * np.log(max(P, 1e-6) / P_REF))
    D = D / (1.0 + beta * (dp / DP_REF))
    return D


def engineer_row(T: float, P: float, dp: float, mat_label: int) -> np.ndarray:
    """Create a feature vector matching the training schema."""
    return np.array([[
        T,
        P,
        dp,
        mat_label,
        1.0 / T,
        T ** 2,
        np.log(max(P, 1e-6)),
        T * np.log(max(P, 1e-6)),
    ]])


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)


def arrhenius_curve(mat: str, P: float, dp: float) -> tuple:
    """Return T-array and corresponding theoretical D for a smooth curve."""
    p = MATERIAL_PARAMS[mat]
    T = np.linspace(300, 700, 300)
    D = p["D0"] * np.exp(-p["Ea"] / (R * T))
    D = D * (1.0 + p["alpha"] * np.log(max(P, 1e-6) / P_REF))
    D = D / (1.0 + p["beta"] * (dp / DP_REF))
    return T, D


# ─────────────────────────────────────────────────────────────────────────────
#  STREAMLIT LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="H₂ Diffusion Predictor",
    page_icon="⚗️",
    layout="wide",
)

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { color: #e0e0e0; font-size: 1.9rem; margin: 0; }
    .main-header p  { color: #a0c4ff; margin: 0.4rem 0 0; font-size: 1rem; }
    .metric-card {
        background: #f8f9fa;
        border-left: 5px solid #0f3460;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
    }
    .metric-label { font-size: 0.85rem; color: #555; font-weight: 600; }
    .metric-value { font-size: 1.4rem; font-weight: 700; color: #0f3460; }
    .error-pct    { font-size: 1.4rem; font-weight: 700; }
</style>
<div class="main-header">
    <h1>⚗️ Hydrogen Diffusion in Metal Hydrides — ML Predictor</h1>
    <p>Mass Transfer Operations · PBL Laboratory Project</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Operating Conditions")

    material = st.selectbox(
        "Metal Hydride",
        list(MATERIAL_PARAMS.keys()),
        help="Select the metal hydride material"
    )

    temperature = st.slider(
        "Temperature (K)", min_value=300, max_value=700,
        value=450, step=5,
        help="Operating temperature in Kelvin"
    )

    pressure = st.slider(
        "Pressure (bar)", min_value=1.0, max_value=50.0,
        value=10.0, step=0.5,
        help="Hydrogen gas pressure in bar"
    )

    particle_size = st.slider(
        "Particle Size (µm)", min_value=10.0, max_value=200.0,
        value=50.0, step=5.0,
        help="Average particle diameter in micrometres"
    )

    st.markdown("---")
    st.markdown("**Theory Basis**")
    st.latex(r"D(T) = D_0 \, e^{-E_a / RT}")
    params_disp = MATERIAL_PARAMS[material]
    st.markdown(f"""
    | Parameter | Value |
    |-----------|-------|
    | D₀ | {params_disp['D0']:.2e} m²/s |
    | Eₐ | {params_disp['Ea']/1000:.0f} kJ/mol |
    """)

# ── Main Predictions ──────────────────────────────────────────────────────────
model = load_model()
df_data = load_dataset()

mat_label = MATERIAL_PARAMS[material]["label"]
D_theory  = theoretical_diffusivity(temperature, pressure, particle_size, material)

col1, col2, col3 = st.columns(3)

if model is None:
    st.warning(
        "⚠️ No trained model found. Please run `python train_model.py` first.\n\n"
        "Showing theoretical values only."
    )
    D_predicted = D_theory
    pct_error   = 0.0
else:
    X_new = engineer_row(temperature, pressure, particle_size, mat_label)
    ln_D_pred   = model.predict(X_new)[0]
    D_predicted = np.exp(ln_D_pred)
    pct_error   = abs(D_predicted - D_theory) / D_theory * 100.0

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🤖 ML Predicted Diffusivity</div>
        <div class="metric-value">{D_predicted:.4e} m²/s</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">📐 Theoretical Diffusivity (Arrhenius)</div>
        <div class="metric-value">{D_theory:.4e} m²/s</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    err_color = "#27ae60" if pct_error < 5 else ("#f39c12" if pct_error < 15 else "#e74c3c")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">📊 Percentage Error</div>
        <div class="error-pct" style="color:{err_color}">{pct_error:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

# ── Arrhenius Curve + User Point ──────────────────────────────────────────────
st.subheader(f"Arrhenius Diffusion Curve — {material}")
fig, ax = plt.subplots(figsize=(10, 4.5))
T_curve, D_curve = arrhenius_curve(material, pressure, particle_size)
color = MAT_COLORS[material]

ax.semilogy(T_curve, D_curve, color=color, lw=2.5, label="Theoretical")
ax.semilogy(temperature, D_theory, "s", color="navy",
            ms=10, zorder=5, label="Theory @ input")
if model is not None:
    ax.semilogy(temperature, D_predicted, "^", color="#e74c3c",
                ms=10, zorder=5, label="ML Prediction @ input")

ax.set_xlabel("Temperature (K)", fontsize=12)
ax.set_ylabel("Diffusivity (m²/s)", fontsize=12)
ax.set_title(f"{material}  |  P = {pressure} bar  |  dₚ = {particle_size} µm",
             fontsize=12)
ax.legend()
ax.grid(True, which="both", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
st.pyplot(fig)
plt.close()

# ── Dataset overview ──────────────────────────────────────────────────────────
if df_data is not None:
    with st.expander("📄 Dataset Preview (first 50 rows)"):
        st.dataframe(df_data.head(50), use_container_width=True)

    with st.expander("📊 Dataset Statistics"):
        st.dataframe(df_data.describe(), use_container_width=True)

# ── Theory Reference ──────────────────────────────────────────────────────────
with st.expander("📚 Mass Transfer Theory Reference"):
    st.markdown("""
    ### Fick's First Law
    Describes steady-state diffusion flux:
    $$J = -D \\frac{dC}{dx}$$

    ### Fick's Second Law
    Describes transient diffusion:
    $$\\frac{\\partial C}{\\partial t} = D \\nabla^2 C$$

    ### Arrhenius Equation for Diffusivity
    $$D(T) = D_0 \\, e^{-E_a / RT}$$
    - $D_0$ = pre-exponential factor (m²/s)
    - $E_a$ = activation energy (J/mol)
    - $R$ = 8.314 J/(mol·K)
    - $T$ = temperature (K)

    ### Effective Diffusivity with Pressure Correction
    $$D_{eff} = D(T) \\cdot \\left(1 + \\alpha \\ln\\frac{P}{P_{ref}}\\right)$$

    ### Particle Size Effect
    $$D_{eff} = D(T) \\bigg/ \\left(1 + \\beta \\frac{d_p}{d_{p,ref}}\\right)$$
    """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Mass Transfer Operations · PBL Laboratory · "
    "Chemical Engineering · Built with Streamlit + scikit-learn"
)
