# House Price Predictions using ML

This project demonstrates house price prediction using the **California Housing dataset** and the **XGBoost Regressor**. The workflow involves data exploration, visualization, model training, and evaluation using **R-squared error** and **Mean Absolute Error (MAE)**.

---

## Project Overview

The goal of this project is to predict house prices based on various features in the California Housing dataset. Key steps include:
- Loading and understanding the dataset.
- Performing Exploratory Data Analysis (EDA) to visualize feature relationships.
- Training a regression model using **XGBoost Regressor**.
- Evaluating the model performance to measure accuracy.

---

## Features

1. **Dataset**: California Housing dataset from `sklearn.datasets`.
2. **EDA**:
   - Visualize correlations using a heatmap.
   - Gain insights into feature importance and data trends.
3. **Model Training**:
   - Utilizes **XGBoost Regressor**, a scalable machine learning model.
4. **Evaluation**:
   - Calculates **R-squared error** to assess accuracy.
   - Computes **Mean Absolute Error (MAE)** for error quantification.

---

## Installation

To run this project locally, follow these steps:

1. Clone this repository:
   ```bash
   gh repo clone TNWRx/House-Prices-Predictions-using-ML
   ```
2. Navigate to the project directory:
   ```bash
   cd House-Prices-Predictions-using-ML
   ```
3. Install the required Python libraries:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn xgboost
   ```

---

## Usage

1. Load the California Housing dataset using `sklearn.datasets.fetch_california_housing`.
2. Perform EDA, including generating a heatmap for feature correlations.
3. Train the **XGBoost Regressor** on the dataset.
4. Evaluate the model using **R-squared** and **MAE** metrics.

To execute the script, run:
```bash
python house_price_prediction.py
```

---

## Results

- **R-squared Error**: Measures the proportion of variance explained by the model.
- **Mean Absolute Error (MAE)**: Quantifies the average prediction error.

Performance metrics and visualizations will be displayed after running the script.

---

## Example Code

```python
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()

# Data preprocessing and visualization
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Price'] = data.target

# Heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Split data
X = df.drop(['Price'], axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBRegressor()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("R-squared:", metrics.r2_score(y_test, y_pred))
print("MAE:", metrics.mean_absolute_error(y_test, y_pred))
```

---

## Contributing

Contributions are welcome! Feel free to fork the repository, create a feature branch, and submit a pull request.

---

## License

This project is licensed under the Apache License. See the `LICENSE` file for more details.

--- 
