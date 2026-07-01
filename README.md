# Retail-Customer-Churn-Prediction
ML classification model predicting customer churn from retail transaction data, with an AI-powered risk assistant.
## Context

A classification model that identifies high-risk customers likely to churn, built from raw retail transaction and demographic data. Designed to help an online retail business proactively target at-risk customers before they leave — maximizing Customer Lifetime Value (CLV) through early intervention.

## Business Problem

Customer churn is a major cost center in online retail — acquiring a new customer is significantly more expensive than retaining an existing one. This project builds a predictive model to flag customers at risk of churning based on their transaction history and demographic profile, so the business can proactively offer retention incentives, personalized outreach, or targeted promotions.

**Hypothesis:** A customer's purchasing behavior (transaction frequency, total spending, recency of last purchase, product category diversity) is a strong predictor of future churn.

**Optimization target:** F1-score and Recall — prioritizing the business's ability to catch as many true churners as possible while maintaining a reasonable precision/recall balance.

## Dataset

- ~23,000 raw transaction records across 16 columns, cleaned to ~20,860 after removing duplicates and negative-quantity (return) entries
- Aggregated into a customer-level dataset using RFM-style features:
  - **Recency** — days since last purchase
  - **Frequency** — total number of transactions
  - **Monetary** — total amount spent
  - **Unique Product Categories** — breadth of product categories purchased
  - **Gender**, **Most Frequent Store Type**
  - **Churn** — binary label, threshold set at 90 days of inactivity

## Approach

1. **Data cleaning & feature engineering** — deduplication, outlier treatment (IQR method), datetime conversion, and construction of RFM-style features from raw transaction logs
2. **Exploratory data analysis** — univariate and bivariate analysis (pair plots, correlation heatmap) to understand churn drivers; Recency emerged as the strongest visual separator between churned and active customers
3. **Modeling** — three classifiers built and compared:
   - Decision Tree (baseline + GridSearchCV-tuned, with class weighting to address class imbalance)
   - K-Nearest Neighbors
   - Logistic Regression
4. **Evaluation** — precision, recall, F1-score, confusion matrices, and ROC/AUC comparison across all three models

## Key Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|---|---|---|---|---|---|
| Decision Tree (baseline) | 0.634 | 0.740 | 0.754 | 0.747 | 0.5435 |
| Decision Tree (GridSearchCV-tuned) | 0.715 | 0.715 | 0.999 | 0.834 | **0.6210** |
| KNN (k=5) | 0.665 | 0.719 | 0.873 | 0.789 | 0.5349 |
| Logistic Regression | 0.716 | 0.716 | **1.000** | 0.834 | 0.5066 |

*(All metrics on held-out test set)*

Logistic Regression achieved perfect recall and tied for the highest F1-score, aligning with the business goal of catching every potential churner. However, its near-random AUC (0.51) and zero specificity revealed this came from classifying almost every customer as a churner rather than genuine discriminatory power. The **tuned Decision Tree** achieved comparable recall (0.999) and F1 (0.834) while showing the best actual separation between churners and non-churners (AUC 0.62), making it the more balanced choice. **Monetary value** and **age at transaction** were the most influential features in the tree's decisions.

## Business Recommendations

- **Proactive re-engagement:** automated alerts as customers approach the 90-day recency threshold, triggering targeted discounts, feedback surveys, or "we miss you" campaigns
- **Cross-category promotion:** customers engaging with fewer product categories churn more — recommend bundles and personalized cross-sell to encourage broader exploration
- **Omnichannel incentives:** customers loyal to a single store type (e.g. e-Shop only) are at higher risk — incentivize channel exploration
- **RFM-based segmentation:** classify customers into Champions, At-Risk, New, and Needs Attention segments for tailored retention strategy

## Bonus: AI-Powered Churn Assistant

The notebook also includes an interactive chatbot layer (Gradio + OpenAI GPT) that lets a user input a customer's profile (frequency, monetary value, age, product categories, store type) and receive a real-time churn risk prediction with a confidence score, plus natural-language insights and a narrative report generated on demand.

## Tools & Libraries

Python, pandas, NumPy, scikit-learn (DecisionTreeClassifier, KNeighborsClassifier, LogisticRegression, GridSearchCV), matplotlib, seaborn, Gradio, OpenAI API

## Files

- `churn_prediction_model.ipynb` — full analysis notebook: data cleaning, EDA, feature engineering, model building, tuning, and evaluation

## Author
Raksha Chabhadia 

MBA - Business Analytics
