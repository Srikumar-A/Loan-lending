# Loan Lending Exploratory Data Analysis & Classification

This project analyzes a loan lending dataset and applies various classification models to predict loan default risk. The workflow includes data exploration, preprocessing, feature engineering, model training, evaluation, and hyperparameter tuning.

## Dataset

The dataset is downloaded from Kaggle using `kagglehub` and contains features such as:
- `credit.policy`: Loan approval status
- `purpose`: Reason for loan
- `int.rate`: Interest rate
- `installment`: Installment amount
- `log.annual.inc`: Log annual income
- `dti`: Debt-to-income ratio
- `fico`: Credit score
- `days.with.cr.line`: Days since oldest credit line
- `revol.bal`: Revolving balance
- `revol.util`: Revolving credit utilization
- `inq.last.6mths`: Loan inquiries in last 6 months
- `delinq.2yrs`: Delinquencies in last 2 years
- `pub.rec`: Public records
- `not.fully.paid`: Target (loan default indicator)

## Workflow

1. **Data Loading & Exploration**
   - Load CSV data using pandas.
   - Explore features, check for nulls, and visualize distributions.

2. **Feature Engineering**
   - One-hot encode the `purpose` categorical feature.
   - Split data into train/test sets.

3. **Model Training & Evaluation**
   - Train and evaluate:
     - Decision Tree (`sklearn.tree.DecisionTreeClassifier`)
     - Random Forest (`sklearn.ensemble.RandomForestClassifier`)
     - XGBoost (`xgboost.XGBClassifier`)
     - CatBoost (`catboost.CatBoostClassifier`)
   - Use accuracy, precision, recall, and F1-score for evaluation.
   - Visualize confusion matrices.

4. **Hyperparameter Tuning**
   - Use GridSearchCV for Random Forest, XGBoost, and CatBoost.
   - Experiment with genetic algorithms for CatBoost hyperparameters.

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- catboost
- kagglehub
- deap

Install dependencies with:

```sh
pip install -r requirements.txt
```

## Usage

Run the notebook [eda.ipynb](eda.ipynb) step by step to reproduce the analysis and modeling.

## Results

- Decision Tree: Baseline performance, visible overfitting.
- Random Forest: Improved accuracy, but recall needs improvement.
- XGBoost: Best accuracy, but requires hyperparameter tuning.
- CatBoost: Good balance, less overfitting, focus on recall.

## Next Steps

- Further optimize hyperparameters using genetic algorithms.
- Address class imbalance (undersampling, SMOTE, etc.).
- Explore neural networks or autoencoders if needed.

## Project Structure

- `eda.ipynb`: Main notebook with code and analysis.
- `catboost_info/`: CatBoost training logs and metrics.
- `requirements.txt`: Python dependencies.

---

**Author:**  
*Your Name