# Viva Questions & Answers
# Machine Learning-Based Prediction of Hydrogen Diffusion in Metal Hydrides

---

## SECTION A — Mass Transfer & Diffusion Theory

**Q1. Define Fick's First Law and explain its physical significance.**

**A:** Fick's First Law states that the molar flux J of a species through a medium is proportional to its concentration gradient:
J = −D · (dC/dx)
The negative sign indicates that diffusion occurs in the direction of decreasing concentration. D (m²/s) is the diffusion coefficient — a material and temperature-specific proportionality constant. It applies under steady-state conditions where the concentration profile does not change with time.

---

**Q2. What is Fick's Second Law and when is it used?**

**A:** Fick's Second Law describes transient (time-dependent) diffusion:
∂C/∂t = D·∇²C
It is used when the concentration profile at a given location changes with time — for example, during the charging or discharging of a metal hydride particle. It reduces to ∂²C/∂x² in one Cartesian dimension and involves a 1/r²·∂/∂r(r²·∂C/∂r) term in spherical coordinates (relevant for spherical hydride particles).

---

**Q3. What is the Arrhenius equation and what does each term represent?**

**A:** D(T) = D₀ · exp(−Eₐ/RT)
- D₀ is the pre-exponential factor or frequency factor (m²/s) — represents the maximum diffusivity at infinite temperature
- Eₐ is the activation energy for diffusion (J/mol) — the energy barrier that hydrogen atoms must overcome to jump between interstitial sites
- R is the universal gas constant (8.314 J/mol·K)
- T is the absolute temperature (K)
A higher Eₐ means D is more sensitive to temperature and smaller at a given T.

---

**Q4. What is an Arrhenius plot and what information can you extract from it?**

**A:** An Arrhenius plot graphs ln(D) on the y-axis against 1/T on the x-axis. From the Arrhenius equation in linearised form:
ln(D) = ln(D₀) − (Eₐ/R)·(1/T)
The plot gives a straight line where:
- Slope = −Eₐ/R → activation energy: Eₐ = −slope × R
- y-intercept = ln(D₀) → pre-exponential factor

---

**Q5. Why does the diffusion coefficient increase with temperature?**

**A:** At higher temperatures, hydrogen atoms possess greater thermal energy (kT). This increases the probability of an atom having enough energy to overcome the activation energy barrier Eₐ between adjacent interstitial sites. The Boltzmann factor exp(−Eₐ/RT) quantitatively captures this: a higher T gives a larger exponential value, increasing D. For MgH₂, D increases approximately 2000× over a 100 K temperature rise (300→400 K).

---

**Q6. What is the difference between steady-state and transient diffusion?**

**A:** In steady-state diffusion, the concentration at every point in the material does not change with time — Fick's First Law applies and there is a constant flux. In transient diffusion, the concentration at each point changes with time as hydrogen gradually penetrates the solid — Fick's Second Law must be used. Metal hydride charging/discharging is inherently transient; once equilibrium is reached (full saturation), the process can be approximated as quasi-steady-state.

---

**Q7. What is the Fourier number in the context of diffusion, and what is its significance?**

**A:** The Fourier number for mass transfer is Fo = Dt/R² (where R is the particle radius). It is a dimensionless measure of how far diffusion has progressed relative to the particle size. When Fo < 0.1, the concentration front has not yet reached the particle core and a surface-layer approximation applies. When Fo > 0.2, the infinite-series solution (Fick's Second Law) converges to a single-term approximation, greatly simplifying calculations.

---

## SECTION B — Metal Hydrides

**Q8. What is a metal hydride and how does hydrogen storage work in it?**

**A:** A metal hydride is a compound formed when hydrogen atoms occupy interstitial sites in a metal or alloy lattice. Hydrogen gas (H₂) dissociates at the metal surface into H atoms (Sieverts' law: C_surface ∝ √P), which then diffuse inward and occupy tetrahedral or octahedral voids in the crystal structure. The process is reversible — applying heat or reducing pressure releases the stored hydrogen. The diffusion step is typically rate-limiting in practical hydrogen uptake/release cycles.

---

**Q9. Why does MgH₂ have a high activation energy compared to LaNi₅H₆?**

**A:** MgH₂ has a high activation energy (~62 kJ/mol) because Mg–H bonds are strongly ionic, and H atoms sit in deep potential wells in the tetragonal rutile structure. The energy required for an H atom to jump to an adjacent interstitial site is therefore large. In contrast, LaNi₅H₆ has a hexagonal CaCu₅ structure with more open interstitial channels. The Ni–H interaction is weaker and more covalent, providing a shallower energy landscape and lower Eₐ (~28 kJ/mol), enabling faster diffusion at the same temperature.

---

**Q10. Why is particle size important in hydrogen diffusion?**

**A:** The characteristic diffusion time scales as τ ~ R²/D. Smaller particles (R → small) have shorter diffusion paths and proportionally faster hydrogen penetration to the core. In practice, ball-milling metal hydrides to nanometric particle sizes dramatically improves absorption/desorption kinetics. In this project, the particle size correction D_eff = D / (1 + β·dp/dp_ref) captures this inverse relationship: larger particles yield a lower effective diffusivity from an engineering standpoint.

---

## SECTION C — Dataset & Modelling

**Q11. Why was a synthetic dataset used instead of real experimental data?**

**A:** Real experimental diffusion data for metal hydrides is scarce, inconsistent across literature sources, and subject to varying measurement conditions. Using physics-based equations (Arrhenius + corrections) allows complete control over the experimental design, full reproducibility, and generation of 1,200 systematically distributed samples that would be prohibitively expensive to collect experimentally. The physics model is validated against published D₀ and Eₐ values from Schlapbach & Züttel (2001) and Grochala & Edwards (2004).

---

**Q12. Why was noise added to the synthetic dataset?**

**A:** Real experimental measurements of diffusion coefficients always contain uncertainty from measurement instruments, sample preparation variability, and impurities. A Gaussian noise level of ±3% was added to simulate this realistic experimental scatter. This makes the ML problem harder and more representative, preventing the model from overfitting to a perfectly smooth mathematical surface.

---

**Q13. Why was the target variable ln(D) instead of D itself?**

**A:** The diffusion coefficient D spans roughly 10 orders of magnitude (10⁻¹⁸ to 10⁻⁸ m²/s) across the three materials and temperature range studied. Training on raw D values would cause the ML model to prioritise large-D samples (LaNi₅H₆ at high T) and completely ignore small-D predictions (MgH₂ at low T). Taking the natural logarithm compresses the range and aligns the target with the inherently multiplicative (exponential) Arrhenius structure. All metrics were then computed in the original D space after exponentiating predictions.

---

**Q14. What is feature engineering and why were features like 1/T added?**

**A:** Feature engineering is the process of creating new input variables from raw features that better represent the underlying physics. The 1/T feature was added because the Arrhenius equation is linear in ln(D) vs 1/T — providing this feature directly gives the linear model access to the correct functional form. T² captures non-linear temperature effects; ln(P) aligns with the logarithmic pressure correction in the model; T·ln(P) captures the interaction between temperature and pressure sensitivity.

---

## SECTION D — Machine Learning

**Q15. What is Random Forest and how does it make predictions?**

**A:** Random Forest is an ensemble method that builds N independent decision trees (N=200 in this project), each trained on a bootstrap sample of the training data and a random subset of features at each split (bagging). Predictions are made by averaging the outputs of all trees. This reduces variance (overfitting) compared to a single deep decision tree. The randomness in feature selection and data sampling decorrelates the trees, improving generalisation. It natively handles non-linear relationships and interaction effects.

---

**Q16. How does Gradient Boosting differ from Random Forest?**

**A:** Random Forest builds trees independently in parallel and averages them (bagging — variance reduction). Gradient Boosting builds trees sequentially in series, where each new tree fits the residuals (errors) of all previous trees combined (boosting — bias reduction). The prediction is a weighted sum: F(x) = F₀ + lr·h₁(x) + lr·h₂(x) + … where lr is the learning rate. Gradient Boosting typically achieves higher accuracy but is more sensitive to hyperparameters and slower to train.

---

**Q17. Explain R², RMSE, and MAE as model evaluation metrics.**

**A:**
- **R² (Coefficient of Determination):** Fraction of variance in D explained by the model. R² = 1 − SS_res/SS_tot. Range: (−∞, 1]. R² = 1 → perfect prediction; R² = 0 → model performs no better than predicting the mean; R² < 0 → model is worse than the mean.
- **RMSE (Root Mean Squared Error):** √(mean of squared errors). Same units as D (m²/s). Penalises large errors more heavily than MAE. Sensitive to outliers.
- **MAE (Mean Absolute Error):** Mean of absolute errors. More robust to outliers than RMSE. Represents the average magnitude of error in m²/s.

For this project: lower RMSE/MAE and higher R² = better model.

---

**Q18. Why did Linear Regression achieve negative R² in test-set evaluation?**

**A:** The model was trained in ln(D) space but evaluated in original D space. Linear Regression in ln(D) has cross-validation R² = 0.96 — it fits the log-transformed data well. However, when predictions are exponentiated back to D space for metric calculation, the error is amplified for samples with high D values (LaNi₅H₆ at high T). The relative error in log-space becomes an absolute error multiplied by D, which is large for high-D samples. The result is that SS_res > SS_tot in original D space, giving R² < 0. Tree-based models avoid this because they are inherently scale-invariant.

---

**Q19. What is cross-validation and why was it used?**

**A:** Cross-validation (5-fold in this project) divides the training data into 5 equal folds. The model is trained 5 times, each time using 4 folds for training and 1 fold for validation. The CV score is the average R² across all 5 runs. This estimates how well the model generalises to unseen data and detects overfitting (large gap between training R² and CV R²). Both Random Forest (CV R² = 0.9997) and Gradient Boosting (0.9998) showed no significant overfitting.

---

**Q20. What does feature importance tell us about the physics?**

**A:** Feature importance in tree-based models measures the average decrease in impurity (MSE) contributed by each feature across all trees. In this project:
1. Temperature (1/T and T combined) accounts for ~60% — confirming that temperature is the dominant variable in the Arrhenius equation.
2. Material label ~20% — reflects the large differences in D₀ and Eₐ between the three hydrides.
3. Particle size ~12% — consistent with the particle-size correction being a secondary effect.
4. Pressure ~8% — matches the weak logarithmic pressure dependence in the model.

This result is physically interpretable and provides confidence that the ML model has learned the underlying physics, not spurious correlations.

---

## SECTION E — Applications & Critical Thinking

**Q21. Can this ML model be used for materials not in the training set?**

**A:** No. The model was trained on MgH₂, TiFeH₂, and LaNi₅H₆ only. Applying it to a new material (e.g., Mg₂FeH₆) would require either (a) retraining with data for that material, or (b) using a transfer learning approach. The model has no fundamental way to extrapolate the physical properties of an unseen material from the material label alone. This is a key limitation — the theoretical Arrhenius equation, by contrast, can be applied to any material if D₀ and Eₐ are known.

---

**Q22. What are the practical limitations of this project?**

**A:**
1. **Synthetic data** — no actual experimental measurements were used; real data would validate whether the assumed Eₐ and D₀ values are accurate for specific material grades.
2. **Simplified pressure model** — the logarithmic correction is empirical; the actual pressure dependence follows the PCT (Pressure-Concentration-Temperature) isotherm, which is more complex.
3. **1D diffusion assumption** — real particles are 3D and may be anisotropic (e.g., MgH₂ has different D along different crystallographic axes).
4. **Isothermal assumption** — real systems experience temperature gradients during charging/discharging.
5. **Training range** — predictions outside 300–700 K and 1–50 bar should not be trusted.

---

**Q23. How would you improve this project with more time?**

**A:** Improvements could include:
- Using experimental D values from published literature (e.g., NIST hydrogen properties database) to validate or replace the synthetic dataset.
- Implementing physics-informed neural networks (PINNs) that embed Fick's Second Law as a constraint in the loss function.
- Extending to 2D/3D diffusion simulation using finite-element methods for non-spherical particles.
- Including microstructural parameters (grain boundary density, crystallite size) that affect D in nanostructured hydrides.
- Adding uncertainty quantification (prediction intervals) using Gaussian Process Regression or Bayesian Neural Networks.

---

**Q24. What is the Fourier number and how would you use it in this project?**

**A:** The Fourier number for mass transfer Fo = Dt/R² is a dimensionless time for diffusion. In this project, for a particle of radius R = 25 µm (dp = 50 µm) and a characteristic time t = 3600 s (1 hour):
- For MgH₂ at 300 K: D ≈ 10⁻¹⁸ m²/s → Fo ≈ 10⁻¹⁸×3600/(625×10⁻¹²) ≈ 5.8×10⁻⁹ (essentially no diffusion)
- For MgH₂ at 600 K: D ≈ 10⁻¹⁰ m²/s → Fo ≈ 576 (complete saturation)
This demonstrates why MgH₂ requires high temperature for practical hydrogen storage — at room temperature, diffusion is negligible on engineering timescales.
