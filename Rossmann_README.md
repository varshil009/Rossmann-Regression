# Rossmann Retail Sales Forecasting

## About the Project

Rossmann operates over 3,000 drug stores across 7 European countries. Store managers are tasked with predicting daily sales up to six weeks in advance. Sales are influenced by factors including promotions, competition, school and state holidays, seasonality, and store locality.

This project builds a regression-based machine learning model to forecast sales for 1,115 Rossmann stores using historical data, helping the business make data-driven decisions on budgets, hiring, incentives, and growth plans.

## Dataset

- **Stores:** 1,115 Rossmann stores across Europe
- **Features:** Store type, assortment, promotions, competition distance, school/state holidays, day of week, and more
- **Target:** Daily sales revenue
- **Download Links:**
  - [Shops Data](https://drive.google.com/file/d/1cjmLnxQQsk_tt74sJWYoefOAFhl2C4bM/view?usp=sharing)
  - [Sales Data](https://drive.google.com/file/d/1NiaSYRPjY5Z2k7rC-HC2bKoFH6Uf7r9f/view?usp=sharing)
  - [ML Ready Data](https://drive.google.com/file/d/1uKUfgEuDePUGWPXLXXS89XD4SOv3n7rg/view?usp=sharing)

## Approach

1. **Exploratory Data Analysis (EDA)** — Analyzed sales trends, seasonality, promotional impact, and store-level patterns
2. **Feature Engineering** — Extracted and transformed features from promotions, competition, holidays, and temporal attributes
3. **Model Benchmarking** — Trained and compared 5 regression models to identify the best performer
4. **Evaluation** — Used MAE, MAPE, and RMSE as evaluation metrics

## Model Comparison

| Model | MAE | MAPE (%) | RMSE |
|-------|-----|----------|------|
| **Random Forest Regressor** | **383.06** | **5.46** | **577.59** |
| SARIMA | 365.87 | 12.66 | 434.03 |
| XGBoost Regressor | 509.39 | 7.27 | 739.63 |
| Linear Regression | 1045.57 | 15.05 | 1458.30 |
| LR Lasso | 1107.31 | 15.66 | 1582.54 |

> **Random Forest Regressor** achieved the best balance of performance with the lowest MAPE of **5.46%**, meaning predictions deviate from actual sales by only ~5.5% on average.

## Best Model Configuration

```python
RandomForestRegressor(
    n_estimators=30,
    random_state=42,
    criterion='gini',
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='auto',
    bootstrap=True,
    oob_score=False,
    class_weight=None
)
```

## Tech Stack

- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Statsmodels (SARIMA), Matplotlib, Seaborn
- **Models:** Random Forest, SARIMA, XGBoost, Linear Regression, Lasso Regression

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/varshil009/Rossmann-Regression.git
   cd Rossmann-Regression
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn xgboost statsmodels matplotlib seaborn
   ```

3. Download the dataset using the links above and place the files in the project directory.

4. Run the notebook:
   ```bash
   jupyter notebook "Rossman Regression.ipynb"
   ```

## Project Structure

```
Rossmann-Regression/
├── ML_process.ipynb            # ML pipeline and model training
├── Rossman Regression.ipynb    # EDA and data preprocessing
└── README.md                   # Project documentation
```

## Key Takeaways

- Random Forest outperformed all other models on MAPE (5.46%), the most business-relevant metric for sales forecasting
- SARIMA achieved the lowest MAE but had a significantly higher MAPE (12.66%), indicating inconsistent percentage-wise accuracy across stores
- Feature engineering on temporal and promotional features was critical to improving model performance
- Forecasting at store level enables targeted business decisions for budgets, staffing, and inventory management
