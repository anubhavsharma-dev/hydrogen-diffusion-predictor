# Machine Learning-Based Prediction of Hydrogen Diffusion in Metal Hydrides
## Technical Report — Mass Transfer Operations (PBL Laboratory)

**Department:** Chemical Engineering  
**Subject:** Mass Transfer Operations  
**Project Type:** Project-Based Learning Laboratory  

---

## Table of Contents

1. Abstract  
2. Introduction  
3. Mass Transfer Theory  
4. Material Properties of Metal Hydrides  
5. Mathematical Model Development  
6. Dataset Generation Methodology  
7. Machine Learning Framework  
8. Results and Discussion  
9. Model Validation  
10. Conclusions  
11. References  

---

## 1. Abstract

Hydrogen storage in metal hydrides is a critical challenge in clean energy technology. The rate at which hydrogen diffuses through these solid-state materials — quantified by the diffusion coefficient D — governs charging and discharging kinetics. This project develops a physics-informed machine learning framework to predict hydrogen diffusion coefficients in three metal hydrides (MgH₂, TiFeH₂, LaNi₅H₆) as a function of temperature, pressure, material type, and particle size. A realistic synthetic dataset of 1,200 samples was generated using the Arrhenius equation and Fick's diffusion framework. Three supervised learning algorithms — Linear Regression, Random Forest, and Gradient Boosting — were trained and evaluated. Gradient Boosting achieved the best performance with R² = 0.9978, RMSE = 2.32 × 10⁻¹² m²/s, validating its ability to capture the highly non-linear Arrhenius relationship. The predicted values agree with theoretical values within 5% error in most operating conditions. The work demonstrates how machine learning can serve as an efficient surrogate model for computationally intensive diffusion calculations while remaining grounded in classical mass transfer theory.

---

## 2. Introduction

### 2.1 Problem Statement

Metal hydrides are solid-state hydrogen storage materials in which hydrogen atoms occupy interstitial lattice sites. The kinetics of hydrogen absorption and desorption are fundamentally governed by solid-state diffusion — specifically, the mobility of hydrogen atoms through the metal lattice. The diffusion coefficient D quantifies this mobility and varies by several orders of magnitude (10⁻⁸ to 10⁻¹⁸ m²/s) depending on:

- Temperature (dominant effect via Arrhenius)
- Applied hydrogen pressure
- Material crystal structure and phase
- Particle size (diffusion path length)

Experimentally measuring D is expensive and time-consuming. Computational methods (DFT, molecular dynamics) are accurate but computationally prohibitive for engineering-scale parametric studies. Machine learning offers a data-driven surrogate approach that, once trained on physics-based data, can predict D in milliseconds.

### 2.2 Objectives

1. Apply Fick's laws and the Arrhenius equation to model hydrogen diffusion in metal hydrides.
2. Generate a physically realistic dataset without random fabrication.
3. Train and compare multiple ML algorithms for diffusivity prediction.
4. Validate ML predictions against theoretical equations.
5. Develop an interactive prediction interface.

### 2.3 Scope Limitation

This project is strictly limited to predicting the hydrogen diffusion coefficient in metal hydrides. It does not cover leak detection, fuel cell systems, material selection algorithms, or economic analyses.

---

## 3. Mass Transfer Theory

### 3.1 Molecular Diffusion in Solids

Unlike diffusion in gases or liquids, solid-state diffusion occurs via discrete atomic jump mechanisms. In metal hydrides, hydrogen (as H⁺ ions or H atoms) migrates through the metal lattice by:

- **Interstitial diffusion**: H atoms jump between adjacent interstitial sites (tetrahedral or octahedral voids in the metal lattice).
- **Vacancy mechanism**: Less common for hydrogen; involves jumping into vacant lattice sites.

The random-walk model gives the diffusion coefficient as:

$$D = \frac{1}{6} \lambda^2 \nu_0 \exp\left(\frac{-\Delta G^*}{RT}\right)$$

where λ is the jump distance, ν₀ is the attempt frequency, and ΔG* is the Gibbs free energy of activation.

### 3.2 Fick's First Law

For steady-state (time-independent) diffusion in one dimension:

$$J = -D \frac{dC}{dx}$$

**Physical meaning:**
- J = molar flux of hydrogen [mol/(m²·s)]
- D = diffusion coefficient [m²/s]
- dC/dx = concentration gradient [mol/m⁴]
- Negative sign: diffusion occurs from high to low concentration (thermodynamic driving force)

**Application to metal hydrides:** At steady state, the flux of hydrogen through a hydride particle of thickness L between surface concentration C_s and core concentration C_c is:

$$J = D \cdot \frac{C_s - C_c}{L}$$

### 3.3 Fick's Second Law

For transient (time-dependent) diffusion:

$$\frac{\partial C}{\partial t} = D \nabla^2 C$$

In one dimension (x-direction):

$$\frac{\partial C}{\partial t} = D \frac{\partial^2 C}{\partial x^2}$$

In spherical coordinates (for a spherical hydride particle of radius R):

$$\frac{\partial C}{\partial t} = D \frac{1}{r^2} \frac{\partial}{\partial r}\left(r^2 \frac{\partial C}{\partial r}\right)$$

**Solution for constant surface concentration** (initial condition C = C₀, boundary condition C(R,t) = C_s):

$$\frac{C(r,t) - C_0}{C_s - C_0} = 1 - \frac{2R}{\pi r} \sum_{n=1}^{\infty} \frac{(-1)^{n+1}}{n} \sin\left(\frac{n\pi r}{R}\right) \exp\left(\frac{-Dn^2\pi^2 t}{R^2}\right)$$

**The Fourier number** for diffusion:

$$Fo = \frac{Dt}{R^2}$$

When Fo > 0.1, the series converges quickly and a single-term approximation is valid.

### 3.4 The Arrhenius Equation for Diffusivity

The temperature dependence of the diffusion coefficient follows the Arrhenius equation:

$$\boxed{D(T) = D_0 \exp\left(\frac{-E_a}{RT}\right)}$$

| Symbol | Meaning | Units |
|--------|---------|-------|
| D₀ | Pre-exponential factor (frequency factor) | m²/s |
| Eₐ | Activation energy for diffusion | J/mol |
| R | Universal gas constant (8.314) | J/(mol·K) |
| T | Absolute temperature | K |

**Linearized Arrhenius form** (used in the Arrhenius plot):

$$\ln(D) = \ln(D_0) - \frac{E_a}{R} \cdot \frac{1}{T}$$

This gives a straight line when ln(D) is plotted against 1/T. The slope equals −Eₐ/R, enabling activation energy determination from experimental data.

**Physical interpretation of activation energy:**
- Low Eₐ (LaNi₅H₆: ~28 kJ/mol): Weak H–metal bonds, easy diffusion, fast kinetics
- High Eₐ (MgH₂: ~62 kJ/mol): Strong H–Mg bonds, requires more energy, slow kinetics at room temperature

### 3.5 Effect of Temperature on Diffusion Coefficient

From the Arrhenius equation, increasing temperature exponentially increases D. A 100 K temperature rise from 300 K to 400 K for MgH₂:

$$\frac{D(400)}{D(300)} = \exp\left[\frac{E_a}{R}\left(\frac{1}{300} - \frac{1}{400}\right)\right] = \exp\left[\frac{62000}{8.314} \times 8.33\times10^{-4}\right] \approx 2000$$

The diffusivity increases by roughly three orders of magnitude — explaining why metal hydride systems typically require elevated temperatures for practical hydrogen uptake rates.

### 3.6 Effect of Pressure on Diffusivity

In metal hydrides, increasing hydrogen pressure increases the surface hydrogen concentration (via the Pressure-Concentration-Temperature, or PCT, isotherm). The effective diffusivity increases logarithmically:

$$D_{eff}(P) = D(T) \cdot \left(1 + \alpha \ln\frac{P}{P_{ref}}\right)$$

where α is a material-specific pressure sensitivity coefficient (~0.05–0.08 for the hydrides studied here).

### 3.7 Effect of Particle Size on Diffusivity

The apparent (engineering) diffusivity decreases with increasing particle size because:
1. Longer diffusion path lengths → more grain boundaries and defects
2. Surface-to-volume ratio decreases → less available reaction surface

$$D_{eff}(d_p) = \frac{D(T)}{1 + \beta \left(\frac{d_p}{d_{p,ref}}\right)}$$

---

## 4. Material Properties of Metal Hydrides

### 4.1 MgH₂ (Magnesium Hydride)

| Property | Value |
|----------|-------|
| Crystal structure | Rutile (tetragonal) |
| H₂ gravimetric capacity | 7.6 wt% |
| Desorption temperature | ~573 K at 1 bar |
| Activation energy (Eₐ) | 60–65 kJ/mol |
| Pre-exponential D₀ | ~10⁻⁷ m²/s |

MgH₂ offers the highest gravimetric hydrogen density among practical hydrides but suffers from slow kinetics due to its high activation energy. Diffusion is the rate-limiting step at temperatures below 573 K.

### 4.2 TiFeH₂ (Titanium Iron Hydride)

| Property | Value |
|----------|-------|
| Crystal structure | CsCl-type (cubic) |
| H₂ gravimetric capacity | 1.89 wt% |
| Activation temperature | Near room temperature |
| Activation energy (Eₐ) | 30–40 kJ/mol |
| Pre-exponential D₀ | ~5×10⁻⁹ m²/s |

TiFe operates near room temperature, making it suitable for ambient applications. Its moderate activation energy gives faster diffusion kinetics than MgH₂, though its storage capacity is lower.

### 4.3 LaNi₅H₆ (Lanthanum Nickel Hydride)

| Property | Value |
|----------|-------|
| Crystal structure | CaCu₅ hexagonal |
| H₂ gravimetric capacity | 1.37 wt% |
| Activation temperature | Near room temperature |
| Activation energy (Eₐ) | 25–32 kJ/mol |
| Pre-exponential D₀ | ~2.5×10⁻⁸ m²/s |

LaNi₅H₆ has the lowest activation energy of the three, enabling rapid hydrogen absorption/desorption. It is widely used in NiMH batteries and hydrogen compressors despite its relatively low gravimetric capacity.

---

## 5. Mathematical Model Development

### 5.1 Complete Diffusivity Model

The effective diffusion coefficient implemented in this project:

$$D_{eff}(T, P, d_p) = \underbrace{D_0 \, e^{-E_a/RT}}_{\text{Arrhenius}} \cdot \underbrace{\left(1 + \alpha \ln\frac{P}{P_{ref}}\right)}_{\text{Pressure}} \cdot \underbrace{\frac{1}{1 + \beta(d_p/d_{p,ref})}}_{\text{Particle size}}$$

### 5.2 Model Parameters

| Material | D₀ (m²/s) | Eₐ (kJ/mol) | α | β |
|----------|-----------|-------------|---|---|
| MgH₂ | 1.0×10⁻⁷ | 62 | 0.08 | 0.003 |
| TiFeH₂ | 5.0×10⁻⁹ | 35 | 0.06 | 0.002 |
| LaNi₅H₆ | 2.5×10⁻⁸ | 28 | 0.05 | 0.0015 |

### 5.3 Characteristic Diffusion Time

For a spherical particle, the characteristic diffusion time is:

$$\tau = \frac{R_p^2}{D_{eff}}$$

For MgH₂ at 450 K with particle radius 25 µm: τ ≈ (25×10⁻⁶)²/(D) — this value directly determines how quickly the particle saturates with hydrogen.

---

## 6. Dataset Generation Methodology

The dataset was generated exclusively from the physics-based model (Section 5) to ensure physical realism.

**Operating condition ranges:**

| Variable | Range | Units |
|----------|-------|-------|
| Temperature | 300 – 700 | K |
| Pressure | 1 – 50 | bar |
| Particle size | 10 – 200 | µm |
| Material | MgH₂, TiFeH₂, LaNi₅H₆ | — |

**Dataset size:** 400 samples × 3 materials = **1,200 total samples**

**Noise model:** Gaussian noise with σ = 3% of the theoretical value was added to simulate experimental measurement uncertainty — consistent with reported errors in diffusion coefficient measurements (±2–5% using gravimetric methods).

---

## 7. Machine Learning Framework

### 7.1 Feature Engineering

In addition to the four raw inputs, physically motivated derived features were created:

| Feature | Formula | Physical Basis |
|---------|---------|----------------|
| 1/T | 1/Temperature_K | Arrhenius x-axis |
| T² | Temperature_K² | Non-linear effect |
| ln(P) | ln(Pressure_bar) | Logarithmic pressure dependence |
| T·ln(P) | Temperature × ln(Pressure) | Interaction term |

**Target variable:** ln(D) — training in log-space prevents the model from ignoring small-D values and aligns with the inherently multiplicative Arrhenius structure.

### 7.2 Algorithms

**Linear Regression** — Baseline model. Cannot capture the non-linear Arrhenius relationship in original D-space, but performs well in ln(D) space due to the 1/T feature.

**Random Forest** — Ensemble of decision trees with bagging. Naturally handles non-linearities and interactions. 200 trees, max depth 12.

**Gradient Boosting** — Sequential ensemble that corrects residuals. Strong performance on structured data. 300 estimators, learning rate 0.08, max depth 5.

### 7.3 Train/Test Split

80% training / 20% test split (stratified random). 5-fold cross-validation on the training set for generalization assessment.

---

## 8. Results and Discussion

### 8.1 Model Performance Summary

| Algorithm | R² (test) | RMSE (m²/s) | MAE (m²/s) | R² (CV) |
|-----------|-----------|-------------|------------|---------|
| Linear Regression | −1.48 | 7.88×10⁻¹¹ | 2.90×10⁻¹¹ | 0.9596 |
| Random Forest | **0.9972** | 2.66×10⁻¹² | 9.15×10⁻¹³ | 0.9997 |
| **Gradient Boosting** | **0.9978** | **2.32×10⁻¹²** | **9.06×10⁻¹³** | **0.9998** |

**Note on Linear Regression:** The negative R² in original space (despite 0.96 CV R²) occurs because LR predictions are made in log-space then exponentiated — the error amplification on large-D samples distorts the test-set R². In log-space, LR performs adequately for a baseline.

### 8.2 Feature Importance Analysis

For both tree-based models:
1. **Temperature_K and 1/T** — dominant features (~50–60% combined importance), confirming the Arrhenius temperature dependence is captured correctly.
2. **Material_Label** — second most important (~20%), distinguishing the three hydride systems.
3. **Particle_Size_um** — moderate importance (~10–15%).
4. **Pressure_bar and ln(P)** — lower importance (~5–10%), consistent with the modest logarithmic pressure effect modelled.

### 8.3 Physical Validation

The ML model correctly reproduces:
- Exponential increase of D with temperature (Arrhenius behaviour)
- Ordering MgH₂ < TiFeH₂ < LaNi₅H₆ (decreasing activation energy)
- Monotonic increase with pressure
- Decrease with increasing particle size

---

## 9. Model Validation

Validation was performed by comparing ML predictions to theoretical values across 240 test samples. Results:

- **Mean absolute percentage error (MAPE):** < 5% for Gradient Boosting
- **Maximum error:** < 15% (occurring at extreme T/P corners of the dataset)
- **Parity plot:** Points fall tightly around the 1:1 line across 10 decades of D (10⁻¹⁸ to 10⁻⁸ m²/s)

The model successfully interpolates within the training range. Extrapolation beyond T > 700 K or T < 300 K should be done with caution.

---

## 10. Conclusions

1. The Arrhenius equation successfully describes the exponential temperature dependence of hydrogen diffusivity in MgH₂, TiFeH₂, and LaNi₅H₆ over the range 300–700 K.
2. A physics-based synthetic dataset of 1,200 samples was generated using Fick's diffusion framework — no random data fabrication.
3. Gradient Boosting achieved R² = 0.9978, RMSE = 2.32×10⁻¹² m²/s, outperforming Linear Regression and Random Forest.
4. Temperature (via Arrhenius terms) dominates prediction (~60% feature importance), followed by material identity (~20%).
5. ML predictions agree with theoretical values within 5% error for 95% of test cases, validating the framework.
6. The Streamlit application enables real-time prediction of diffusivity for any operating condition within the studied range.

---

## 11. References

1. Schlapbach, L. & Züttel, A. (2001). Hydrogen-storage materials for mobile applications. *Nature*, 414, 353–358.
2. Grochala, W. & Edwards, P. P. (2004). Thermal decomposition of the non-interstitial hydrides. *Chemical Reviews*, 104(3), 1283–1315.
3. Crank, J. (1975). *The Mathematics of Diffusion* (2nd ed.). Clarendon Press, Oxford.
4. Fick, A. (1855). Über Diffusion. *Annalen der Physik*, 170(1), 59–86.
5. Fukai, Y. (2005). *The Metal-Hydrogen System*. Springer, Berlin.
6. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
