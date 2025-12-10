# ML_Kaggle_project
# Project Overview
This project focuses on a regression task to predict the Purchase Value (purchaseValue) of user sessions based on digital footprint data (traffic source, device details, geography, and browsing behavior). The goal is to accurately estimate the revenue generated per session to help businesses optimize marketing strategies and user targeting.This solution employs advanced data preprocessing pipelines, feature engineering, and an ensemble modeling approach utilizing XGBoost and Random Forest.
#  Dataset
The dataset contains session-level data similar to Google Analytics schema.

Train Data: ~116k entries.Target Variable: purchaseValue (Continuous).

Note: The target is highly skewed with a large number of zero values (non-conversions).

Key Features: pageViews, totalHits, timeOnSite, trafficSource, geoNetwork, device, operatingSystem, etc.

# Project Workflow
1.Exploratory Data Analysis (EDA)Analyzed correlations between browsing metrics (pageViews, totalHits) and revenue.

Investigated the distribution of the target variable (highly skewed) and totalHits.

Analyzed categorical impact: Mean purchase value by Continent, OS, and Browser.

Identified that Mobile users behaved differently than Desktop users regarding conversion.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

2. Data Preprocessing & PipelineTo prevent data leakage and ensure reproducibility, a robust sklearn.pipeline was implemented:Missing Value Imputation:Numerical features: Median imputation.Categorical features: Mode (most frequent) imputation.Encoding: Ordinal Encoding for categorical variables.Scaling: StandardScaler applied to numerical features.
3. Feature EngineeringCustom features were created to capture user engagement intensity:pageViews_per_session: Page views normalized by session number.log_pageViews: Log transformation to handle skewness.pageViews_to_totalHits: Ratio of views to hits.interaction_pageHits: Interaction term between views and hits.
4. Model Selection & TuningVarious regression models were experimented with:Linear Regression: Baseline model.MLP Regressor (Neural Network): Tested but achieved lower $R^2$.LightGBM: Tested, but underperformed compared to tree ensembles.Random Forest: Strong baseline ($R^2 \approx 0.60$).XGBoost: Best single model performance.Hyperparameter Tuning:Performed RandomizedSearchCV on XGBoost to find optimal parameters:n_estimators: 550max_depth: 11learning_rate: 0.09subsample: 0.75. Ensemble Strategy (Voting Regressor)
The final solution utilizes a Voting Regressor to combine the strengths of multiple models:XGBoost (Tuned): Captures complex non-linear patterns.Random Forest: Reduces variance and overfitting.Meta-Voting: A final layer combining the pipeline results.Final Cross-Validation Score ($R^2$): ~0.5337
# Dependencies
Python 3.xPandas / NumPyMatplotlib / SeabornScikit-LearnXGBoostLightGBM
