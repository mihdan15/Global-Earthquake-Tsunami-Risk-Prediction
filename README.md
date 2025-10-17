# 🌊 Global Earthquake–Tsunami Risk Prediction using Machine Learning

An end-to-end AI project that predicts tsunami potential from global earthquake data (2001–2022).  
Built with **Python**, **Scikit-Learn**, and **Streamlit**, this project demonstrates how machine learning can be applied to disaster risk prediction — from data preprocessing to model deployment.

---

## 📘 Overview
This project analyzes and predicts tsunami potential using the **Global Earthquake–Tsunami Risk Assessment Dataset**, a clean, machine-learning–ready dataset covering **782 major earthquakes worldwide** between 2001–2022.  

It consists of two major parts:
1. **📊 Historical Analysis Mode** – visualize and analyze global earthquake–tsunami distribution (with temporal patterns).
2. **🔮 Prediction Mode** – predict future tsunami potential based on new earthquake parameters (no time-based bias).

---

## 🗂️ Dataset Information
**Source:** Global Earthquake–Tsunami Risk Assessment Dataset (2001–2022)  
**Total Records:** 782  
**Features:** 13  
**Target:** `tsunami` (0 = non-tsunami, 1 = tsunami)

| Feature | Type | Description |
|----------|------|-------------|
| magnitude | float | Earthquake magnitude (Richter scale) |
| depth | float | Focal depth in km |
| sig | int | Significance score |
| mmi | int | Modified Mercalli Intensity |
| cdi | int | Community Decimal Intensity |
| nst | int | Number of seismic stations |
| dmin | float | Distance to nearest station (°) |
| gap | float | Azimuthal gap between stations (°) |
| latitude, longitude | float | Epicenter coordinates |
| Year, Month | int | Temporal data (used for historical analysis only) |
| tsunami | binary | Target variable (1 = tsunami, 0 = non-tsunami) |

---

## 🧹 Data Preprocessing
All preprocessing and analysis were performed in **Google Colab**.

### Steps:
1. **Import & Inspection**
   - Loaded CSV, checked null values, datatypes, and duplicates.
2. **Exploratory Data Analysis (EDA)**
   - Global scatter map (`plotly.express.scatter_geo`)
   - Correlation heatmap
   - Magnitude vs Depth analysis
3. **Feature Engineering**
   - `mag_depth_ratio = magnitude / depth`
   - `depth_category` (shallow / intermediate / deep)
   - One-hot encoding for depth category
4. **Train-Test Split**
   - 80% training, 20% testing
   - Stratified by target class to maintain balance

---

## 🤖 Modeling Process
Used **Scikit-Learn** for model training and evaluation.

### 1️⃣ Logistic Regression (Baseline)
- Scaled features using `StandardScaler`
- Result: ROC-AUC ≈ **0.93**

### 2️⃣ Random Forest Classifier (Default)
- Captured non-linear relationships between features
- Result: ROC-AUC ≈ **0.965**

### 3️⃣ Random Forest (Tuned)
Used `GridSearchCV` with parameters:
```python
{
  "n_estimators": [100, 200, 300],
  "max_depth": [10, 20, None],
  "min_samples_split": [2, 5, 10],
  "min_samples_leaf": [1, 2, 4]
}
```
→ Best ROC-AUC: 0.966

### 4️⃣ Feature Importance 
Top predictive features:
- **magnitude**
- **depth**
- **longitude**
- **latitude**
- **mag_depth_ratio**

---

### 🧠 Model Variants
| Model | Features | Use Case |
|--------|-----------|-----------|
| `tsunami_rf_model.pkl` | includes `Year`, `Month` | for historical pattern analysis |
| `tsunami_rf_noyear.pkl` | excludes temporal features | for future prediction (real-time) |

**Reason:**  
Year-based features improve accuracy for 2001–2022,  
but make the model **biased** toward specific years.  
Therefore, a second “future-ready” model was built **without Year/Month**.

---

## 🌍 Streamlit Application

### 🔧 Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

### 🖥️ App Structure

- `/data/earthquake_data_tsunami.csv`
- `tsunami_rf_model.pkl`
- `tsunami_rf_noyear.pkl`
- `tsunami_features.pkl`
- `tsunami_features_noyear.pkl`
- `app.py`
- `requirements.txt`

## 🖼️ Streamlit App Features

### 📊 1. Historical Analysis Mode
Visualizes historical data and model behavior:
- Global scatter map (Tsunami vs Non-Tsunami)
- Magnitude vs Depth scatter
- Heatmap of high-risk tsunami zones
- Yearly trends and frequency
- Magnitude distribution
- Correlation heatmap of numerical features

### 🔮 2. Prediction Mode
Predicts tsunami likelihood for new earthquake inputs:
- User inputs parameters (magnitude, depth, latitude, etc.)
- Calculates derived features (depth category, mag_depth_ratio)
- Displays:
  - Tsunami Probability (% probability)
  - Prediction Result (Likely / Not Likely)
  - Epicenter Map Visualization

---

## 📊 Visual Insights

| Visualization     | Description                                      |
|-------------------|--------------------------------------------------|
| 🌍 Scatter Geo     | Global distribution of tsunamis and earthquakes |
| 🔥 Heatmap         | Density map of high-risk tsunami areas          |
| 📅 Yearly Trends   | Number and rate of tsunami events per year      |
| 📈 Histogram       | Magnitude distribution per class                |
| 🧩 Correlation     | Relationship between physical features          |
| 🌋 Feature Importance | Key parameters influencing tsunami potential |

---

## 🚀 Key Results
- Final Model ROC-AUC: **0.966**
- Balanced dataset (**39% tsunami events**)
- Predicts tsunami likelihood with high reliability
- Deployed on Streamlit for interactive visualization

---

## 🧭 Tech Stack

| Category        | Tools                                      |
|----------------|---------------------------------------------|
| Data Processing | Pandas, NumPy                              |
| Visualization   | Plotly, Seaborn, Matplotlib                |
| Modeling        | Scikit-Learn (Logistic Regression, Random Forest) |
| Tuning          | GridSearchCV                               |
| Deployment      | Streamlit                                  |
| Environment     | Google Colab, Python 3.10                  |

---

## 🧩 Project Structure

```kotlin
project/
│
├── data/
│   └── earthquake_data_tsunami.csv
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_model_evaluation.ipynb
│
├── models/
│   ├── tsunami_rf_model.pkl
│   └── tsunami_rf_noyear.pkl
│
├── app.py
├── requirements.txt
└── README.md
```


## 🧠 Insights & Learnings
- Temporal features (Year) can introduce bias in prediction — removed for generalization.
- Shallow, high-magnitude earthquakes near coasts are most tsunami-prone.
- Combining physical and spatial parameters improves accuracy.
- Streamlit enables intuitive storytelling of machine learning workflows.


