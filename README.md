# 🛒 Engage 2 Value: From Clicks to Conversions

## 📌 Project Overview
This regression project aims to predict the **Purchase Value** (`purchaseValue`) of user sessions based on digital footprint data. By analyzing traffic sources, device details, geography, and browsing behavior, the model estimates the revenue generated per session. This insight assists businesses in optimizing marketing strategies and improving user targeting.

The final solution employs an ensemble modeling approach, utilizing a **Voting Regressor** that combines **XGBoost** and **Random Forest** models, supported by robust preprocessing and feature engineering pipelines.

---

## 📂 Dataset Details
The dataset follows a schema similar to Google Analytics, capturing session-level interactions.

* **Training Data:** ~116,000 entries.
* **Target Variable:** `purchaseValue` (Continuous).
    * *Note:* The target is highly skewed, containing a significant number of zero values (indicating non-conversions).
* **Key Features:**
    * `pageViews` & `totalHits`
    * `timeOnSite`
    * `trafficSource`
    * `geoNetwork`
    * `device` & `operatingSystem`

---

## 🛠️ Project Workflow

### 1. Exploratory Data Analysis (EDA)
Comprehensive analysis was performed to understand data characteristics:
* **Correlation Analysis:** Examined relationships between browsing metrics (`pageViews`, `totalHits`) and revenue.
* **Distribution Analysis:** Investigated the highly skewed target variable and `totalHits`.
* **Categorical Impact:** Analyzed mean purchase value across different Continents, Operating Systems (OS), and Browsers.
* **Device Behavior:** Identified distinct conversion behavior differences between Mobile and Desktop users.

### 2. Data Preprocessing & Pipeline
To prevent data leakage and ensure reproducibility, a robust `sklearn.pipeline` was implemented.

* **Missing Value Imputation:**
    * *Numerical features:* Median imputation.
    * *Categorical features:* Mode (most frequent) imputation.
* **Encoding:** Ordinal Encoding applied to categorical variables.
* **Scaling:** `StandardScaler` applied to numerical features.

### 3. Feature Engineering
Custom features were engineered to capture the intensity of user engagement:

| Feature Name | Description |
| :--- | :--- |
| `pageViews_per_session` | Page views normalized by the session number. |
| `log_pageViews` | Logarithmic transformation to handle skewness in page view data. |
| `pageViews_to_totalHits` | The ratio of views to total hits. |
| `interaction_pageHits` | An interaction term capturing the relationship between views and hits. |

### 4. Model Selection & Tuning
Various regression models were experimented with to establish a baseline and find the optimal predictor:

* **Linear Regression:** Used as the baseline model.
* **MLP Regressor (Neural Network):** Tested but yielded a lower $R^2$.
* **LightGBM:** Underperformed compared to other tree-based ensembles in this specific context.
* **Random Forest:** Provided a strong baseline performance ($R^2 \approx 0.60$).
* **XGBoost:** Achieved the best single-model performance.

#### Hyperparameter Tuning
`RandomizedSearchCV` was performed on the XGBoost model to identify optimal parameters:
* **n_estimators:** 550
* **max_depth:** 11
* **learning_rate:** 0.09
* **subsample:** 0.7

### 5. Ensemble Strategy


The final solution leverages a **Voting Regressor** to aggregate predictions and balance the strengths of the top-performing models:
* **XGBoost (Tuned):** Excellent at capturing complex, non-linear patterns.
* **Random Forest:** Effective at reducing variance and preventing overfitting.
* **Meta-Voting:** A final layer combining the results of the pipeline.

### 🏆 Final Results
* **Cross-Validation Score ($R^2$):** ~0.5337

---

## 📦 Dependencies
To reproduce this project, the following Python libraries are required:

* Python 3.x
* Pandas / NumPy
* Matplotlib / Seaborn
* Scikit-Learn
* XGBoost
* LightGBM
