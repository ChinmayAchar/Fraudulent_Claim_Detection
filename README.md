🛡️ Insurance Claim Fraud Detection
Authors: Chinmay H R & Shashank Vinod Chardeve

Project Type: Machine Learning / Classification

Status: Completed

📌 Project Overview
Insurance fraud is a significant issue leading to massive financial losses and increased premiums. This project implements a machine learning pipeline to automate the detection of fraudulent insurance claims. By analyzing historical claims data, the system identifies patterns, anomalies, and risk factors to flag suspicious activities for manual review.



📂 Repository Structure
Fraudulent_Claim_Detection_Starter_Chinmay_H_R_Shashank_Vinod_Chardeve.ipynb: The main Jupyter Notebook containing the end-to-end ML pipeline (Data cleaning, EDA, Feature Engineering, Modeling).

Fraudulent_Claim_Detection...pdf: Project documentation and report.

README.md: Project summary and instructions.

🛠️ Technologies & Libraries
The project is implemented in Python using the following libraries:

Data Manipulation: pandas, numpy

Visualization: matplotlib, seaborn

Machine Learning: scikit-learn (sklearn)

Modules: model_selection, preprocessing, linear_model, ensemble, metrics, feature_selection

Resampling: imblearn (specifically RandomOverSampler)

⚙️ Project Pipeline
1. Data Preprocessing & Cleaning
The raw data undergoes rigorous cleaning to prepare it for analysis:


Handling Missing Values: Null values (often represented as ?) are identified and imputed or removed.



Data Type Conversion: Date fields (e.g., policy_bind_date, incident_date) are converted to datetime objects for time-series analysis.


Dropping Columns: Irrelevant ID columns and superfluous features are removed to reduce noise.


Splitting: The data is split into training and validation sets using a Stratified 70-30 split to maintain class distribution.

2. Exploratory Data Analysis (EDA)
Comprehensive analysis to understand data distribution:


Univariate Analysis: Histograms and stats for key numerical features.


Bivariate Analysis: Boxplots and likelihood tables to observe relationships between categorical variables and fraud.


Correlation: Heatmaps to identify collinearity among features.


Imbalance Detection: Identified a class imbalance: 75% Non-Fraud vs 25% Fraud.

3. Feature Engineering
Techniques used to improve model predictive power:


Class Balancing: Applied Random Over Sampler (ROS) to handle the class imbalance.


Feature Extraction: Created new features, such as the duration (in days) between the policy binding date and the incident date.


Encoding: Consolidated rare categories and applied Dummy/One-Hot Encoding for categorical variables.


Scaling: Applied standard scaling to numerical features.

4. Model Building & Evaluation
Two primary classification models were developed and optimized.

A. Logistic Regression

Feature Selection: Recursive Feature Elimination with Cross-Validation (RFECV) was used, validated by p-values and VIF checks.

Optimization: The classification threshold was tuned to maximize sensitivity (recall) for fraud detection.

Initial Threshold (0.5): Sensitivity = 0.42


Optimized Threshold (0.2): Sensitivity = 0.94, Accuracy = 86%, F1 Score = 0.87.

B. Random Forest Classifier
Implementation: Used for its robustness against overfitting and ability to handle non-linear relationships.

Tuning: Hyperparameter tuning was performed to maximize accuracy.


Performance: Achieved 92% Accuracy in cross-validation with a Training F1 score of 1.00.



Feature Importance: Identified Total Claim Amount, Incident Timing, and Policy Duration as top drivers of fraud.

📊 Key Findings
The analysis revealed several strong indicators of fraud:


Timing: Suspicious claims often occur shortly after a policy is bound.


Amounts: Disproportionately high claim amounts relative to policy limits are red flags.


Severity Mismatch: Claims reporting minor incidents but claiming major damage (high severity) are often fraudulent.


Demographics: Specific correlations were found with certain occupations and hobbies (e.g., cross-fit, chess).

🚀 Future Improvements

Text Mining: Apply NLP to claim narratives to find inconsistencies in incident descriptions.


Dynamic Retraining: Continuously retrain models with new fraud patterns to prevent model drift.# Fraudulent_Claim_Detection
