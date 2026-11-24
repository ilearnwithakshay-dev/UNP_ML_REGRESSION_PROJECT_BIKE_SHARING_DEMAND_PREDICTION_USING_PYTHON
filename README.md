# Bike Sharing Demand Prediction: A Comparative Regression Study of Environmental vs. Temporal Features

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-yellowgreen.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Accelerated-red.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## Project Links & Access

### Interactive Notebooks (Google Colab)

* **Experiment 1 (Without Time Features):**[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1800h101aT2DNNdCSzoav9X71MxkTdaGV/view?usp=drive_link)
  *Colab URL:* `https://drive.google.com/file/d/1800h101aT2DNNdCSzoav9X71MxkTdaGV/view?usp=drive_link`
* **Experiment 2 (With Time Features Included):**[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1bBhJHLVptSAxCX-_5DucOoM1o9px_uNq/view?usp=sharing)
  *Colab URL:* `https://drive.google.com/file/d/1bBhJHLVptSAxCX-_5DucOoM1o9px_uNq/view?usp=sharing`

### Project Resources & Drive Links

* **Project Presentation Link:** `https://docs.google.com/presentation/d/1CjCoF6cNSWOWWH0upkRualUhTqFJuoOO/edit?usp=drive_link&ouid=108531411860157531016&rtpof=true&sd=true`
* **Dataset Access Link:** `https://drive.google.com/file/d/1kI6vKBja6ZG0Zg5P3u50vUcIDoKgbN_Z/view?usp=drive_link`
* **Dataset Details Link:** `https://drive.google.com/file/d/1Eo4W1Z2QZAVixAfbQ1euY3w5778C3fiP/view?usp=drive_link`

---

## Project Overview

I have completed this Bike Sharing Demand Prediction regression project using Python in collaboration with **United Network of Professionals (UNP)** as part of my data science internship.

The primary objective of this project is to model and forecast hourly bike rental demand using environmental, seasonal, and temporal covariates from the Capital Bikeshare system in Washington D.C. (17,379 hourly observations across 2011 and 2012).

A critical research question explored in this work is: **To what degree do environmental and weather signals alone determine ridership demand, and what is the quantitative impact of incorporating temporal features (such as hour of day, month, and day of week)?**

To address this, two comprehensive experiments were designed, trained, and benchmarked across 8 machine learning regression algorithms and deep Artificial Neural Networks (ANN):

1. **Experiment 1 (`BikeSharing_G6.ipynb`)**: Models trained **strictly on environmental and weather attributes** (temperature, humidity, windspeed, weather condition, season), deliberately excluding all time-based signals.
2. **Experiment 2 (`BikeSharing_Time_G6.ipynb`)**: Models trained on **both environmental and engineered temporal attributes** (hour dummy encoding, month dummy encoding, day of week, working day, year).

The steps, methodology, and experimental results are presented below.

---

## Team & Institutional Affiliation

* **Academic Institution:** Bhavan's Vivekananda College
* **Program:** Data Science Internship Program
* **Collaborating Organization:** United Network of Professionals (UNP)
* **Project Group:** Group 6
* **Team Members:**
  * **Akhiranand Boddani**
  * **Uppaluru S V N Santhosha Roopa**
  * **Deepthi Hansika**
  * **Akshay Kumar**

---

## Dataset Description

The dataset corresponds to two-year historical logs (2011–2012) from the **Capital Bikeshare System** in Washington, D.C., aggregated hourly and enriched with Freemeteo meteorological data.

* **Total Records:** 17,379 hourly observations
* **Target Variable:** `cnt` (Total hourly bike rentals: $cnt = casual + registered$)
* **Predictor Groups:**
  * **Temporal Features:** `season` (1: Spring, 2: Summer, 3: Fall, 4: Winter), `yr` (0: 2011, 1: 2012), `mnth` (1–12), `hr` (0–23), `holiday` (binary), `weekday` (0–6), `workingday` (binary).
  * **Weather/Environmental Features:** `weathersit` (1: Clear/Few clouds, 2: Mist/Cloudy, 3: Light Rain/Snow, 4: Heavy Rain/Ice Pellets), `temp` (Normalized temperature in Celsius, $/41$), `atemp` (Normalized feeling temperature in Celsius, $/50$), `hum` (Normalized humidity, $/100$), `windspeed` (Normalized wind speed, $/67$).

### Data Hygiene & Leakage Prevention

* **Target Leakage Elimination:** The columns `casual` and `registered` were dropped prior to modeling because their sum directly equals `cnt`. Keeping them would represent 100% data leakage.
* **Identifier & Date Pruning:** `instant` (row index) and `dteday` (date string) were removed as they carry no generalizable predictive relationship.
* **Multicollinearity Elimination:** `atemp` (feeling temperature) exhibits near-perfect collinearity with `temp` ($r \approx 0.99$). `atemp` was dropped to avoid coefficient instability.
* **Redundancy Pruning:** `holiday` was dropped in temporal modeling as it is redundant when `workingday` and `weekday` are present.

---

## Project Methodology & Step-by-Step Execution

```
┌─────────────────────────────────┐
│     Raw Data (hour.csv)         │
│      17,379 Hourly Records      │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│     Data Cleaning & Hygiene     │
│  - Drop casual, registered      │
│  - Drop instant, dteday, atemp  │
│  - Target transform: log1p(cnt) │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   Outlier Detection via IQR     │
│   - windspeed, hum, log_cnt     │
│   - 512 outliers dropped (2.9%) │
│   - 16,867 Cleaned Records      │
└────────────────┬────────────────┘
                 │
        ┌────────┴──────────────────────────┐
        ▼                                   ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐
│    Experiment 1: NO TIME      │   │   Experiment 2: WITH TIME     │
│  - Only Weather & Season      │   │  - Weather + Temporal signals │
│  - Dummy encode season/weather│   │  - Dummy encode hr, mnth, etc.│
│  - 9 Predictor Features       │   │  - ~74-75 Predictor Features  │
└───────────────┬───────────────┘   └───────────────┬───────────────┘
                │                                   │
                ▼                                   ▼
┌───────────────────────────────────────────────────────────────────┐
│              StandardScaler & 5-Fold Split Suite                  │
│       Splits: 80/20, 75/25, 70/30, 65/35, 60/40                   │
└─────────────────────────────────┬─────────────────────────────────┘
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────┐
│                   8 Classical ML Regressors                       │
│     Linear Regression, KNN, Linear SVM, Decision Tree,            │
│     Random Forest, AdaBoost, Gradient Boosting, XGBoost           │
│                                                                   │
│                 GridSearchCV 5-Fold Optimization                  │
└─────────────────────────────────┬─────────────────────────────────┘
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────┐
│            Artificial Neural Network (ANN) Sweep                  │
│     Architecture, Optimizer (Adam vs. SGD), Epoch Tuning          │
└─────────────────────────────────┬─────────────────────────────────┘
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────┐
│               Evaluation & Comparative Analysis                   │
│         Metrics: R², MAE, MSE, RMSE, MAPE (%)                     │
└───────────────────────────────────────────────────────────────────┘
```

### Step 1: Data Inspection & Missing Value Audit

* Verified dataset dimensions and confirmed that **0 missing/null values** exist across all 17 columns.
* Data types were converted to suitable formats for numerical scaling and categorical dummy encoding.

### Step 2: Target Variable Transformation

* An exploratory analysis of `cnt` revealed severe positive right-skewness (skewness $> 1.2$).
* Standard linear models assume homoscedasticity and normally distributed residuals. Thus, a logarithmic transformation was applied:
  $$
  \text{log\_cnt} = \ln(cnt + 1) = \text{np.log1p}(cnt)
  $$
* Applying $\log(cnt+1)$ compresses extreme demand spikes, stabilizes variance, transforms the target into a near-normal distribution, and ensures predictions never map to unrealistic negative counts when back-transformed via $\exp(\hat{y}) - 1$.

### Step 3: Outlier Detection and Removal via IQR

* Interquartile Range (IQR) analysis ($Q_1 - 1.5 \times \text{IQR}$ to $Q_3 + 1.5 \times \text{IQR}$) was performed on numerical features:
  * `temp`: 0 outliers detected
  * `hum`: 22 extreme low-humidity outliers
  * `windspeed`: 342 extreme high-wind outliers
  * `log_cnt`: 158 extreme low-count outliers
* **Total rows removed:** 512 outliers (only 2.9% of the dataset), refining the sample from **17,379 to 16,867 clean records**.
* Distribution checks via Kernel Density Estimation (KDE) and Box Plots confirmed improved distributional symmetry and reduced kurtosis.

### Step 4: Correlation & Multicollinearity Analysis (VIF)

* Evaluated Karl Pearson correlation coefficients across numerical attributes and $\Phi_K$ (Phi-K) non-linear correlation across categorical variables.
* Calculated **Variance Inflation Factor (VIF)** with an added constant intercept to ensure no harmful multicollinearity among input variables:
  * `season_SPRING`: 3.75
  * `temp`: 3.06
  * `season_WINTER`: 2.63
  * `season_SUMMER`: 1.87
  * `hum`: 1.60
  * `weathersit_Snow/Rain`: 1.25
  * `weathersit_Mist/Cloudy`: 1.16
  * `windspeed`: 1.13
  * `workingday`: 1.01
* **Result:** All predictor VIF values are comfortably below the threshold of 5.0 (and 10.0), proving coefficient estimation stability.

### Step 5: Feature Engineering & Encoding

* **Experiment 1 (Without Time):**
  * Temporal attributes (`yr`, `mnth`, `hr`, `holiday`, `weekday`) were dropped.
  * `season` and `weathersit` were dummy encoded using `pd.get_dummies(..., drop_first=True)`.
  * Resulting feature matrix: **9 features** (`temp`, `hum`, `windspeed`, `workingday`, encoded seasons, encoded weather).
* **Experiment 2 (With Time):**
  * Retained year `yr` as a binary flag (0: 2011, 1: 2012).
  * One-hot dummy encoded `hr` (1–23, dropping `hr_0`), `mnth` (2–12, dropping `mnth_1`), `weekday`, `workingday`, `season`, and `weathersit`.
  * Resulting feature matrix: **74–75 features**, capturing intra-day commuter peaks, weekend shifts, and yearly growth.

### Step 6: Multi-Split Validation & Feature Scaling

* To eliminate arbitrary partition bias, all baseline models were evaluated across **5 distinct train/test splits**:
  * 80/20, 75/25, 70/30, 65/35, and 60/40.
* Feature scaling was conducted using `StandardScaler`. The scaler was strictly fitted on the training split and transformed on the test split to prevent data leakage.

### Step 7: Machine Learning Model Benchmarking

Eight regression algorithms were systematically benchmarked on both datasets:

1. **Linear Regression (Ordinary Least Squares)**
2. **K-Nearest Neighbors Regressor (KNN)**
3. **Support Vector Regressor (Linear SVR)**
4. **Decision Tree Regressor**
5. **Random Forest Regressor**
6. **AdaBoost Regressor**
7. **Gradient Boosting Regressor**
8. **Extreme Gradient Boosting Regressor (XGBoost)**

### Step 8: Hyperparameter Optimization (GridSearchCV)

* Applied 5-Fold Cross Validation (`cv=5`) on the 80/20 split optimizing Mean Squared Error (`neg_mean_squared_error`).
* Explored hyperparameters including:
  * **KNN:** `n_neighbors` ($[3, 5, 7, 9, 11]$), `weights` (`'uniform'`, `'distance'`).
  * **Linear SVR:** `C` ($[0.01, 0.1, 1, 10]$), `epsilon` ($[0.0, 0.1, 0.2]$).
  * **Decision Tree:** `max_depth` ($[3, 5, 10, \text{None}]$), `min_samples_split`, `min_samples_leaf`.
  * **Random Forest:** `n_estimators` ($[50, 100, 150]$), `max_depth` ($[5, 10, 20]$).
  * **AdaBoost:** `n_estimators` ($[50, 100]$), `learning_rate` ($[0.01, 0.1, 1.0]$).
  * **Gradient Boosting:** `n_estimators` ($[100, 150]$), `learning_rate` ($[0.05, 0.1]$), `max_depth` ($[3, 5]$).
  * **XGBoost:** `n_estimators` ($[100, 150, 200]$), `max_depth` ($[3, 5, 7]$), `learning_rate` ($[0.05, 0.1, 0.2]$).

### Step 9: Deep Learning Architecture Sweeps (ANN)

* Built using `TensorFlow/Keras` with Mean Squared Error loss.
* **Architectures Evaluated:**
  * Shallow: $Input \to 10 \to 5 \to 1$
  * Medium: $Input \to 64 \to 32 \to 16 \to 1$
  * Deep Pyramid: $Input \to 32 \to 32 \to 16 \to 16 \to 8 \to 1$
  * Very Deep: $Input \to 128 \to 64 \to 32 \to 16 \to 8 \to 1$
* **Optimizer Benchmarks:** Adam vs. Stochastic Gradient Descent (SGD) evaluated at 20, 50, 100, and 150 epochs.
* Training and validation loss curves were plotted to monitor convergence and detect overfitting.

---

## Key Insights & Discussion

1. **The Temporal Dominance Finding:**

   * Weather signals alone provide an upper ceiling of $R^2 \approx 0.33$ to $0.35$. Temperature and weather explain baseline seasonal shifts, but cannot differentiate between high-commute peak hours (8 AM, 5 PM) and off-peak hours (3 AM).
   * Incorporating dummy-encoded hour of the day (`hr`) and calendar features accounts for diurnal human mobility routines, unlocking $R^2 > 0.93$.
2. **Decision Tree Overfitting:**

   * Unconstrained Decision Trees achieved a training $R^2$ of 1.00 but performed poorly on test folds ($R^2 = -0.06$ on no-time data). Tree pruning via `GridSearchCV` successfully recovered generalization.
3. **Gradient Boosting vs. Deep Neural Networks:**

   * On low-dimensional environmental data (9 features), tree-based ensembles (XGBoost and Random Forest) outperformed neural networks.
   * On high-dimensional dummy-encoded temporal data (74 features), the deep multi-layer ANN with SGD optimizer achieved the highest overall score ($R^2 = 0.9368$), closely followed by XGBoost ($R^2 = 0.9324$).

---

## Future Scope

* **Real-Time Weather API Integration:** Connecting live forecast endpoints (e.g., OpenWeatherMap, Freemeteo) for automated rolling demand inference.
* **Sequential Deep Learning (LSTM / GRU / Transformers):** Formulating the prediction as an explicit autoregressive sequential time series rather than static tabular regression.
* **Geospatial & Station-Level Modeling:** Extending prediction from system-wide counts to station-specific pickup and dropoff docks to optimize physical bike redistribution logistics.

---

## Acknowledgments

* **United Network of Professionals (UNP):** For providing internship guidance, project mentoring, and structured technical support.
* **Bhavans Vivekananda College:** For academic encouragement and project enablement.
