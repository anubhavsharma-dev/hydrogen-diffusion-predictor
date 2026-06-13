# Machine Learning-Based Prediction of Hydrogen Diffusion in Metal Hydrides
## Mass Transfer Operations — PBL Laboratory

---

## Folder Structure

```
hydrogen_diffusion_project/
│
├── data/
│   ├── generate_dataset.py            ← Physics-based dataset generator
│   └── hydrogen_diffusion_dataset.csv ← Generated dataset (1200 samples)
│
├── models/
│   ├── best_model.pkl                 ← Saved best ML model (Gradient Boosting)
│   └── all_models.pkl                 ← All three trained models
│
├── plots/
│   ├── 01_temperature_vs_diffusivity.png
│   ├── 02_arrhenius_plot.png
│   ├── 03_pressure_vs_diffusivity.png
│   ├── 04_actual_vs_predicted.png
│   ├── 05_feature_importance.png
│   ├── 06_correlation_matrix.png
│   └── 07_model_comparison.png
│
├── reports/
│   ├── technical_report.md            ← Full technical report (Word-ready)
│   ├── ppt_structure.md               ← Presentation slide structure
│   └── viva_questions_answers.md      ← 24 Viva Q&As
│
├── app.py                             ← Streamlit web application
├── train_model.py                     ← ML training + visualisation pipeline
├── requirements.txt                   ← Python dependencies
└── README.md                          ← This file
```

---

## Quick Start

### Step 1 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Generate Dataset
```bash
python data/generate_dataset.py
```
Creates `hydrogen_diffusion_dataset.csv` (1,200 physics-based samples).

### Step 3 — Train Models & Generate Plots
```bash
python train_model.py
```
Trains all three models, saves best model to `models/`, generates all 7 plots to `plots/`.

### Step 4 — Launch Streamlit App
```bash
streamlit run app.py
```
Opens interactive predictor at http://localhost:8501

---

## Model Results Summary

| Model | R² (test) | RMSE (m²/s) | MAE (m²/s) |
|-------|-----------|-------------|------------|
| Linear Regression | −1.48 | 7.88×10⁻¹¹ | 2.90×10⁻¹¹ |
| Random Forest | 0.9972 | 2.66×10⁻¹² | 9.15×10⁻¹³ |
| **Gradient Boosting** ✓ | **0.9978** | **2.32×10⁻¹²** | **9.06×10⁻¹³** |

**Best model: Gradient Boosting** (R² = 0.9978, CV R² = 0.9998)

---

## Physics Background

The diffusion coefficient is computed as:

```
D_eff = D0 × exp(-Ea / RT)          [Arrhenius]
      × (1 + α × ln(P/P_ref))        [Pressure correction]
      ÷ (1 + β × dp/dp_ref)          [Particle size correction]
```

Material parameters from literature:

| Material | D₀ (m²/s) | Eₐ (kJ/mol) |
|----------|-----------|-------------|
| MgH₂ | 1.0×10⁻⁷ | 62 |
| TiFeH₂ | 5.0×10⁻⁹ | 35 |
| LaNi₅H₆ | 2.5×10⁻⁸ | 28 |

---

## Python Version
Python ≥ 3.9 recommended.
