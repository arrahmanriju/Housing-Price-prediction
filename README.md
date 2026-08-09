# California Housing Prices — Regression Project

A machine learning project that predicts median house values for California block groups using the classic California Housing dataset. The notebook walks through the full workflow: exploratory data analysis, data cleaning, feature engineering, model training, evaluation, and a simple interactive prediction demo.

## Dataset

- **File:** `housing.csv`
- **Rows / Columns:** 20,640 rows × 10 columns
- **Target variable:** `median_house_value`
- **Features:** `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`, `ocean_proximity` (categorical)
- **Missing values:** `total_bedrooms` had 207 missing entries, imputed with the column median.

## Project Workflow

### 1. Setup & First Look
Loaded the data into a pandas DataFrame and inspected shape, data types, summary statistics, and sample rows.

### 2. Exploratory Data Analysis (EDA)
- Histograms of all numeric columns
- Geographic scatter plot of `longitude` vs `latitude`, colored by `median_house_value` (recreates the shape of California)
- Correlation matrix / heatmap of numeric features
- Box plot of `median_house_value` by `ocean_proximity`

**Key findings:**
- `median_income` is the strongest predictor of house value (correlation ≈ 0.69)
- Coastal categories (`NEAR BAY`, `<1H OCEAN`, `NEAR OCEAN`) command higher prices than `INLAND`
- House values cluster geographically along the coast and around major cities

### 3. Data Cleaning & Preprocessing
- Filled missing `total_bedrooms` values with the median
- One-hot encoded the `ocean_proximity` categorical column
- Engineered new features:
  - `rooms_per_household`
  - `bedrooms_per_room`
  - `population_per_household`
- Train/test split: 80/20 (`X_train`: 16,512 rows, `X_test`: 4,128 rows)
- Scaled numeric features with `StandardScaler`

### 4. Model Training
Two regression models were trained and compared:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 50,888.66 | 72,668.54 | 0.60 |
| Random Forest Regressor | 32,318.76 | 50,368.16 | **0.81** |

**Random Forest Regressor performed best**, explaining ~81% of the variance in house prices with a notably lower average error than Linear Regression.

### 5. Evaluation
- MAE, RMSE, and R² were used (accuracy/confusion matrix don't apply to regression tasks)
- Predicted vs. Actual scatter plots were generated for both models to visually assess fit

### 6. Prediction Demo
An interactive input cell lets a user enter feature values for a hypothetical block group (location, income, room counts, etc.) and returns a predicted median house value.

## Tech Stack

- **Language:** Python
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn
- **Models:** `LinearRegression`, `RandomForestRegressor` (scikit-learn)

## How to Run

1. Clone this repository / open the notebook in Jupyter or Google Colab.
2. Ensure `housing.csv` is available in the working directory (update the file path in the notebook if needed).
3. Install dependencies:
```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
```
4. Run the notebook cells in order, from data loading through the prediction demo.

## Results Summary

The Random Forest Regressor clearly outperformed the Linear Regression baseline on all three metrics (MAE, RMSE, R²), confirming that the relationship between features and house price is non-linear and benefits from an ensemble tree-based approach.

## Possible Next Steps

- Hyperparameter tuning for Random Forest (e.g., `GridSearchCV`)
- Try additional models (Gradient Boosting, XGBoost)
- Address multicollinearity between engineered features
- Deploy the trained model behind a simple API or web app
