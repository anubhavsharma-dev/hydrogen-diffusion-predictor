# PowerPoint Presentation Structure
# Machine Learning-Based Prediction of Hydrogen Diffusion in Metal Hydrides

## SLIDE 1 — Title Slide
**Title:** Machine Learning-Based Prediction of Hydrogen Diffusion in Metal Hydrides
**Subtitle:** Mass Transfer Operations — PBL Laboratory Project
**Visual:** Atomic lattice of a metal hydride with hydrogen atoms highlighted (blue spheres at interstitial sites)
**Elements:**
- Group member names, Roll numbers
- Department of Chemical Engineering
- Date of presentation

---

## SLIDE 2 — Table of Contents
A clean timeline/roadmap showing the 10 sections with icons:
🔬 Theory → ⚗️ Materials → 📐 Model → 📊 Data → 🤖 ML → 📈 Results → ✅ Validation → 🖥️ App → 💡 Conclusions → ❓ Q&A

---

## SLIDE 3 — Problem Statement & Motivation
**Title:** Why Predict H₂ Diffusivity?
**Left panel:** Problem
- Hydrogen storage in metal hydrides requires fast charge/discharge
- Rate-limiting step is SOLID-STATE DIFFUSION
- Experimental measurement: expensive, slow, material-intensive

**Right panel:** Solution
- Physics-based simulation → Realistic dataset
- Machine Learning → Fast surrogate predictor
- Streamlit App → Instant engineering predictions

**Visual:** Schematic of H₂ gas → Metal hydride particle → H atoms diffusing inward

---

## SLIDE 4 — Fick's Laws of Diffusion
**Title:** The Foundation: Fick's Laws

**Left box — Fick's First Law (Steady State):**
$$J = -D \frac{dC}{dx}$$
Diagram: Linear concentration profile across hydride slab with flux arrow

**Right box — Fick's Second Law (Transient):**
$$\frac{\partial C}{\partial t} = D \nabla^2 C$$
Diagram: Time-evolving concentration profile in spherical particle

**Bottom bar:** Key insight — Both laws require knowing D(T, P, material)

---

## SLIDE 5 — Arrhenius Equation
**Title:** Temperature Dependence of Diffusivity — Arrhenius Equation

**Centre equation (large, highlighted):**
$$D(T) = D_0 \, e^{-E_a/RT}$$

**Parameter table (3 columns):**
| Material | D₀ (m²/s) | Eₐ (kJ/mol) |
|----------|-----------|-------------|
| MgH₂ | 10⁻⁷ | 62 |
| TiFeH₂ | 5×10⁻⁹ | 35 |
| LaNi₅H₆ | 2.5×10⁻⁸ | 28 |

**Right visual:** Arrhenius plot (ln D vs 1/T) showing three straight lines with different slopes

**Take-away callout:** "Slope = −Eₐ/R  →  Steeper slope = higher activation energy = slower diffusion"

---

## SLIDE 6 — Metal Hydrides: Materials Overview
**Title:** Three Metal Hydrides Studied

**Three cards (one per material), each containing:**
- Crystal structure diagram (schematic)
- Colour-coded badge: 🔴 MgH₂ | 🟢 TiFeH₂ | 🟡 LaNi₅H₆
- Key specs: capacity, Eₐ, operating temperature range
- Engineering significance (1–2 lines)

**Bottom comparison bar:** Kinetics: LaNi₅H₆ (fastest) → TiFeH₂ → MgH₂ (slowest)

---

## SLIDE 7 — Mathematical Model & Dataset Generation
**Title:** Physics-Based Dataset Generation (No Random Data)

**Left: Model equations**
1. Arrhenius: D₀·exp(−Eₐ/RT)
2. Pressure correction: × (1 + α·ln(P/P_ref))
3. Particle size: ÷ (1 + β·dp/dp_ref)
4. Gaussian noise: ±3% experimental uncertainty

**Right: Dataset summary table**
| Parameter | Range |
|-----------|-------|
| Temperature | 300–700 K |
| Pressure | 1–50 bar |
| Particle size | 10–200 µm |
| Materials | 3 |
| Total samples | 1,200 |

**Visual:** Flowchart: Equations → Parameters → Dataset (1200 rows)

---

## SLIDE 8 — ML Pipeline
**Title:** Machine Learning Pipeline

**Horizontal flowchart (5 stages):**
1. Raw Dataset (1200 samples)
2. Feature Engineering (8 features incl. 1/T, ln P)
3. Train/Test Split (80/20)
4. Model Training (LR | RF | GB)
5. Evaluation (R², RMSE, MAE)

**Feature engineering box (highlighted):**
Original: T, P, dp, Material
Derived: 1/T, T², ln(P), T·ln(P)
Target: ln(D) — log space prediction

---

## SLIDE 9 — Model Performance Results
**Title:** Results: All Three Models Compared

**Large table (centre):**
| Model | R² (test) | RMSE | MAE | CV R² |
|-------|-----------|------|-----|-------|
| Linear Regression | −1.48 | 7.88×10⁻¹¹ | — | 0.96 |
| Random Forest | 0.9972 | 2.66×10⁻¹² | 9.15×10⁻¹³ | 0.9997 |
| **Gradient Boosting** ✓ | **0.9978** | **2.32×10⁻¹²** | **9.06×10⁻¹³** | **0.9998** |

**Visual:** Model comparison bar chart (from Plot 07)

**Callout banner:** "Gradient Boosting selected as Best Model — R² = 0.9978"

---

## SLIDE 10 — Scientific Visualisations (split across 2 slides if needed)
**Slide 10a: Temperature & Pressure Effects**
- Left: Temperature vs Diffusivity (Arrhenius curves for 3 materials) — Plot 01
- Right: Arrhenius plot ln(D) vs 1000/T showing linear relationship — Plot 02

**Slide 10b: ML Predictions**
- Left: Actual vs Predicted parity plot — Plot 04
- Right: Feature Importance (Gradient Boosting) — Plot 05

---

## SLIDE 11 — Correlation Matrix & Physical Insights
**Title:** Feature Correlations & Physical Interpretation
**Visual:** Correlation heatmap (Plot 06) — full slide
**Annotations highlighting:**
- Strong negative correlation: 1/T ↔ ln(D) = Arrhenius
- Moderate positive: Temperature ↔ ln(D)
- Weak: Pressure ↔ ln(D) (as expected from logarithmic model)

---

## SLIDE 12 — Streamlit Application Demo
**Title:** Interactive Diffusivity Predictor — Streamlit App

**Left: Screenshot** of app sidebar with sliders (T, P, dp, Material)
**Centre: Results panel** showing ML prediction, Theoretical value, % error
**Right: Live curve** showing Arrhenius plot with user point highlighted

**Steps to run:**
```
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

---

## SLIDE 13 — Validation Against Theory
**Title:** ML vs Theory — Validation Results

**Three-column layout:**
| Metric | Value |
|--------|-------|
| MAPE | < 5% |
| Max error | < 15% |
| Samples within 5% | > 95% |

**Visual:** Scatter plot — ML prediction vs theoretical D (parity with ±10% bands)

**Key message:** ML reproduces Arrhenius behaviour, material ranking, and pressure effects correctly.

---

## SLIDE 14 — Conclusions
**Title:** Conclusions

Six numbered points:
1. Arrhenius equation models H₂ diffusivity across 300–700 K with strong agreement
2. Physics-based 1,200-sample dataset generated — no random fabrication
3. Gradient Boosting best performer: R² = 0.9978, RMSE = 2.32×10⁻¹² m²/s
4. Temperature (via 1/T) is the dominant predictor (60% importance)
5. ML predictions agree with theory within 5% for 95% of test cases
6. Streamlit app delivers real-time D prediction for any operating condition

---

## SLIDE 15 — Future Work & Limitations
**Two columns:**

**Limitations:**
- Dataset is synthetic (physics-based, not experimental)
- Model valid only within training range (300–700 K, 1–50 bar)
- Pressure effect simplified to single logarithmic term

**Future Work:**
- Integrate experimental data from literature databases (NIST, HSP)
- Add more materials (NaAlH₄, Mg₂FeH₆, etc.)
- Incorporate 2D/3D Fick's equation for particle geometry effects
- Deploy as a web API for engineering teams

---

## SLIDE 16 — Thank You / Q&A
**Title:** Thank You

Large visual: Periodic table section showing Mg, Ti, Fe, La, Ni highlighted

**Contact / References listed**

"Questions?"
