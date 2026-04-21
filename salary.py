# -*- coding: utf-8 -*-
"""Salary Prediction with Multiple Regression Models"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

# Load the data
df = pd.read_csv('Salary_Data.csv')

# Handle missing values
for column in df.columns:
    if df[column].dtype in ['int64', 'float64']:
        # Fill numerical columns with mean
        df[column] = df[column].fillna(df[column].mean())
    elif df[column].dtype == 'object':
        # Fill categorical columns with mode
        df[column] = df[column].fillna(df[column].mode()[0])

print("Null values after imputation:")
print(df.isnull().sum())

# Check for duplicate rows
duplicate_rows = df[df.duplicated()]
print("Number of duplicate rows:", duplicate_rows.shape[0])

if not duplicate_rows.empty:
    print("Duplicate rows found:")
    print(duplicate_rows)  # Changed from display() to print()
else:
    print("No duplicate rows found.")

# Apply Label Encoding to categorical columns
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

print("DataFrame after Label Encoding:")
print(df.head())  # Changed from display() to print()

# Separate independent and dependent variables
X = df.drop('Salary', axis=1)  # Features
y = df['Salary']  # Target variable

print("Shape of independent variables (X):", X.shape)
print("Shape of dependent variable (y):", y.shape)
print("\nFirst 5 rows of X:")
print(X.head())  # Changed from display() to print()
print("\nFirst 5 values of y:")
print(y.head())  # Changed from display() to print()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

# Linear Regression
linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train, y_train)
y_pred = linear_reg_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nLinear Regression Model Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# Decision Tree Regression
decision_tree_model = DecisionTreeRegressor(random_state=42)
decision_tree_model.fit(X_train, y_train)
y_pred_dt = decision_tree_model.predict(X_test)

mae_dt = mean_absolute_error(y_test, y_pred_dt)
mse_dt = mean_squared_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mse_dt)
r2_dt = r2_score(y_test, y_pred_dt)

print("\nDecision Tree Regressor Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae_dt:.2f}")
print(f"Mean Squared Error (MSE): {mse_dt:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_dt:.2f}")
print(f"R-squared (R2): {r2_dt:.2f}")

# Random Forest Regression
random_forest_model = RandomForestRegressor(random_state=42)
random_forest_model.fit(X_train, y_train)
y_pred_rf = random_forest_model.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("\nRandom Forest Regressor Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae_rf:.2f}")
print(f"Mean Squared Error (MSE): {mse_rf:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_rf:.2f}")
print(f"R-squared (R2): {r2_rf:.2f}")

# Support Vector Regression (SVR)
svr_model = make_pipeline(StandardScaler(), SVR(kernel='rbf'))
svr_model.fit(X_train, y_train)
y_pred_svr = svr_model.predict(X_test)

mae_svr = mean_absolute_error(y_test, y_pred_svr)
mse_svr = mean_squared_error(y_test, y_pred_svr)
rmse_svr = np.sqrt(mse_svr)
r2_svr = r2_score(y_test, y_pred_svr)

print("\nSupport Vector Regressor (SVR) Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae_svr:.2f}")
print(f"Mean Squared Error (MSE): {mse_svr:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_svr:.2f}")
print(f"R-squared (R2): {r2_svr:.2f}")

# K-Nearest Neighbors (KNN) Regression
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

mae_knn = mean_absolute_error(y_test, y_pred_knn)
mse_knn = mean_squared_error(y_test, y_pred_knn)
rmse_knn = np.sqrt(mse_knn)
r2_knn = r2_score(y_test, y_pred_knn)

print("\nK-Nearest Neighbors (KNN) Regressor Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae_knn:.2f}")
print(f"Mean Squared Error (MSE): {mse_knn:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_knn:.2f}")
print(f"R-squared (R2): {r2_knn:.2f}")

# Save the best model (Random Forest Regressor) to a .pkl file
filename = 'best_model.pkl'
pickle.dump(random_forest_model, open(filename, 'wb'))
print(f"\nBest model (Random Forest Regressor) saved to {filename}")

# Create comparison table
metrics_data = {
    'Model': ['Linear Regression', 'Decision Tree', 'Random Forest', 'SVR', 'KNN'],
    'MAE': [mae, mae_dt, mae_rf, mae_svr, mae_knn],
    'MSE': [mse, mse_dt, mse_rf, mse_svr, mse_knn],
    'RMSE': [rmse, rmse_dt, rmse_rf, rmse_svr, rmse_knn],
    'R2 Score': [r2, r2_dt, r2_rf, r2_svr, r2_knn]
}

metrics_df = pd.DataFrame(metrics_data)

# Display the metrics table
print("\nModel Evaluation Metrics Comparison:")
print(metrics_df.round(2).to_string())

# Plotting the metrics
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Regression Model Performance Comparison', fontsize=16)

sns.barplot(x='Model', y='MAE', data=metrics_df, ax=axes[0, 0], hue='Model', palette='viridis', legend=False)
axes[0, 0].set_title('Mean Absolute Error (MAE)')
axes[0, 0].tick_params(axis='x', rotation=45)

sns.barplot(x='Model', y='MSE', data=metrics_df, ax=axes[0, 1], hue='Model', palette='viridis', legend=False)
axes[0, 1].set_title('Mean Squared Error (MSE)')
axes[0, 1].tick_params(axis='x', rotation=45)

sns.barplot(x='Model', y='RMSE', data=metrics_df, ax=axes[1, 0], hue='Model', palette='viridis', legend=False)
axes[1, 0].set_title('Root Mean Squared Error (RMSE)')
axes[1, 0].tick_params(axis='x', rotation=45)

sns.barplot(x='Model', y='R2 Score', data=metrics_df, ax=axes[1, 1], hue='Model', palette='viridis', legend=False)
axes[1, 1].set_title('R-squared (R2 Score)')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
