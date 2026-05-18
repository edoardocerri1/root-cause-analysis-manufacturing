# 🔍 Steel Coil Defect Prediction

> **Real Industry Project** | Bologna Business School — ML Fieldwork Program  
> Predictive model to identify surface defects in steel coil manufacturing using PLC sensor data

---

## 📌 Project Overview

Steel production generates large volumes of sensor data across the manufacturing process. Traditional inspection methods are reactive and limited. This project builds a **machine learning pipeline** to detect defects early and support real-time decision-making.

| | |
|---|---|
| **Dataset** | 299,384 PLC sensor readings · 1,261 unique coils · 106 sensor channels |
| **Target** | 6 binary defect types (Tipo 1–6) across coil metre ranges |
| **Main challenge** | Joining exact sensor positions with defect label ranges — without data leakage |
| **Business goal** | Reduce waste, improve process control and product quality |

---

## 🛠️ ML Pipeline
### 1. EDA & Data Merge
- Position range join: matched PLC rows to defect segments using `MT_FROM <= MT <= MT_TO`
- Key findings: tail-out effect handled with 3 binary indicator columns; AIR_CH4 negatives clipped to zero
- 21 flagged coils removed → final dataset: **298,851 rows**

### 2. Feature Engineering
- 120 initial features → grouped into **10 semantic groups** (Temperature Zones, Pyrometers, Laser, Air, Gas, Pressure, Ventilation, Cooling)
- PCA within each group (95% variance retained) → 30 features
- SelectKBest → **59 final features** (80.56% variance retained)

### 3. Model — XGBoost + GroupKFold
- GroupKFold split **by coil** to avoid data leakage
- Hyperparameter tuning: `n_estimators=150`, `max_depth=5`, `learning_rate=0.1`, `subsample=0.8`
- Handles class imbalance natively; robust to correlated features

### 4. Unsupervised Clustering — Reverse Engineering Defect Labels
Since no data dictionary existed, K-Means (K=6) was applied on coil-level sensor aggregates to reverse-engineer the physical meaning of each defect type.  
**Finding**: defect labels were assigned by visual inspection, not derived from sensor data — confirmed independently by the client.

---

## 📊 Results

| Target | Mean Accuracy | Mean ROC-AUC | Lift @top-20% |
|---|---|---|---|
| DIF_TIPO_3 | 83.1% | 0.887 | 2.62x |
| DIF_TIPO_4 | 85.2% | 0.832 | 2.72x |
| ANY_DEFECT | 81.1% | 0.881 | 2.13x |

> For Defects 3 and 4, the model captures **~60% of defective coils in the top 20%** of ranked predictions — nearly **3x better than random guessing**.

---

## 🔑 Key Insights (SHAP Analysis)

Two sensors consistently distinguish high-risk from low-risk coils:

- **`LASER_RAFF_1`** — strip position post-furnace: wrong exit height → surface defects more likely
- **`AIR_Z5 / GAS_Z5`** — zone 5 furnace combustion: too hot → strip surface damaged before cooling

> *"When the strip exits at the wrong position AND zone 5 runs hotter than usual, defect risk doubles. These two sensors are the earliest warning signs in the entire production line."*  
> Confirmed independently by both clustering and SHAP analysis.

---

## 💡 Proposed Solution

Re-design the current process with two data streams:
1. **PLC Sensor Data** (existing) — real-time risk scoring
2. **Computer Vision camera** (new) — auto-labelling defects at line exit

→ Model retrained continuously → **Live Dashboard** for operators

---

## 🧰 Tech Stack

`Python` `XGBoost` `SHAP` `Scikit-learn` `Pandas` `NumPy` `Matplotlib` `Seaborn` `GroupKFold` `PCA` `K-Means`

---

## 📁 Repository Structure
├── FINAL_metal_coil_defect_analysis.ipynb   # Full analysis notebook
├── presentation/                             # Project presentation slides
└── README.md

## 👤 Authors

**Edoardo Cerri**, Israa Ismail, Giovanni Duca, Filippo Iacchia, Anastasija Smiljkovska  
MSc Data Science & Business Analytics — Bologna Business School
