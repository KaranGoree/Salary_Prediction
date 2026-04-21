# Salary Prediction Machine Learning Project

## Project Description
This project aims to build and evaluate various machine learning regression models to predict salaries based on a given dataset. The workflow includes data loading, comprehensive preprocessing (handling missing values, encoding categorical features), splitting data for training and testing, and applying multiple regression algorithms. The performance of these models is then rigorously evaluated and compared to identify the best-performing solution.

## Dataset
The dataset used for this project is `Salary_Data.csv`, which contains information relevant to salary prediction.

## Key Features
- Data Loading and Initial Exploration
- Missing Value Imputation (Mean for numerical, Mode for categorical)
- Categorical Feature Encoding (Label Encoding)
- Data Splitting (Training and Testing sets)
- Multiple Regression Model Training (Linear Regression, Decision Tree, Random Forest, SVR, KNN)
- Model Evaluation using MAE, MSE, RMSE, and R2 Score
- Visual Comparison of Model Performances
- Saving the Best Model

## Installation
To run this project, you'll need Python and the following libraries. You can install them using pip:

```bash
pip install pandas scikit-learn matplotlib seaborn numpy
```

## Usage
1.  **Clone the repository (if on GitHub):**
    ```bash
    git clone <your-repo-link>
    cd <your-repo-name>
    ```
2.  **Ensure `Salary_Data.csv` is in your project directory.**
3.  **Run the Jupyter Notebook or Colab Notebook:**
    Open the notebook file (`<your_notebook_name>.ipynb`) in a Jupyter environment or Google Colab and run all cells sequentially.
    The notebook will:
    - Load the data.
    - Preprocess the data.
    - Train and evaluate various regression models.
    - Display performance metrics and comparative plots.
    - Save the best model (`best_model.pkl`).

## Models Evaluated
-   **Linear Regression**
-   **Decision Tree Regressor**
-   **Random Forest Regressor**
-   **Support Vector Regressor (SVR)**
-   **K-Nearest Neighbors (KNN) Regressor**

## Results
After evaluating the models, the following R2 Scores were observed:
-   Linear Regression: 0.67
-   Decision Tree Regressor: 0.96
-   Random Forest Regressor: **0.98** (Best Performing)
-   SVR: 0.01
-   KNN Regressor: 0.96

**Random Forest Regressor** was identified as the best model for this dataset due to its highest R2 score and lowest error metrics.

## Best Model
The best-performing model, `RandomForestRegressor`, is saved as `best_model.pkl` using Python's `pickle` library.
