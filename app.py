# -*- coding: utf-8 -*-
import streamlit as st
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

# Set page configuration
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💰",
    layout="wide"
)

# Title and description
st.title("💰 Salary Prediction Model Comparison")
st.markdown("This app compares multiple regression models to predict salaries based on various features.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('Salary_Data.csv')
    return df

try:
    df = load_data()
    st.success("✅ Data loaded successfully!")
    
    # Display raw data
    with st.expander("📊 View Raw Data"):
        st.dataframe(df)
        st.write(f"Dataset Shape: {df.shape}")
    
    # Store original categorical values for later use
    categorical_mappings = {}
    original_categories = {}
    
    # Data preprocessing
    st.header("🔧 Data Preprocessing")
    
    # Handle missing values
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:
            df[column] = df[column].fillna(df[column].mean())
        elif df[column].dtype == 'object':
            df[column] = df[column].fillna(df[column].mode()[0])
    
    st.write("✅ Missing values handled")
    
    # Check duplicates
    duplicate_rows = df[df.duplicated()]
    st.write(f"Number of duplicate rows: {duplicate_rows.shape[0]}")
    
    # Store original categorical values and create label encoders
    categorical_cols = df.select_dtypes(include='object').columns.tolist()  # Convert to list
    label_encoders = {}
    
    for col in categorical_cols:
        # Store unique values for later use in UI
        original_categories[col] = df[col].unique().tolist()
        
        # Create and fit label encoder
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        
        # Store mapping for display
        mapping_dict = dict(zip(le.classes_, le.transform(le.classes_)))
        categorical_mappings[col] = mapping_dict
    
    if categorical_cols:
        st.write(f"✅ Categorical variables encoded: {', '.join(categorical_cols)}")
        with st.expander("View Encoding Mappings"):
            for col, mapping in categorical_mappings.items():
                st.write(f"**{col}:**")
                for original, encoded in mapping.items():
                    st.write(f"  {original} → {encoded}")
    
    # Prepare features and target
    X = df.drop('Salary', axis=1)
    y = df['Salary']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    st.write(f"Training set size: {X_train.shape[0]} samples")
    st.write(f"Testing set size: {X_test.shape[0]} samples")
    
    # Model training and evaluation
    st.header("🤖 Model Training and Comparison")
    
    # Create a placeholder for progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Dictionary to store models and their metrics
    models = {}
    metrics_list = []
    
    # 1. Linear Regression
    status_text.text("Training Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    
    models['Linear Regression'] = lr_model
    metrics_list.append({
        'Model': 'Linear Regression',
        'MAE': mean_absolute_error(y_test, y_pred_lr),
        'MSE': mean_squared_error(y_test, y_pred_lr),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
        'R2 Score': r2_score(y_test, y_pred_lr)
    })
    progress_bar.progress(20)
    
    # 2. Decision Tree
    status_text.text("Training Decision Tree...")
    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    
    models['Decision Tree'] = dt_model
    metrics_list.append({
        'Model': 'Decision Tree',
        'MAE': mean_absolute_error(y_test, y_pred_dt),
        'MSE': mean_squared_error(y_test, y_pred_dt),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_dt)),
        'R2 Score': r2_score(y_test, y_pred_dt)
    })
    progress_bar.progress(40)
    
    # 3. Random Forest
    status_text.text("Training Random Forest...")
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    
    models['Random Forest'] = rf_model
    metrics_list.append({
        'Model': 'Random Forest',
        'MAE': mean_absolute_error(y_test, y_pred_rf),
        'MSE': mean_squared_error(y_test, y_pred_rf),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        'R2 Score': r2_score(y_test, y_pred_rf)
    })
    progress_bar.progress(60)
    
    # 4. SVR
    status_text.text("Training Support Vector Regressor...")
    svr_model = make_pipeline(StandardScaler(), SVR(kernel='rbf'))
    svr_model.fit(X_train, y_train)
    y_pred_svr = svr_model.predict(X_test)
    
    models['SVR'] = svr_model
    metrics_list.append({
        'Model': 'SVR',
        'MAE': mean_absolute_error(y_test, y_pred_svr),
        'MSE': mean_squared_error(y_test, y_pred_svr),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_svr)),
        'R2 Score': r2_score(y_test, y_pred_svr)
    })
    progress_bar.progress(80)
    
    # 5. KNN
    status_text.text("Training K-Nearest Neighbors...")
    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)
    
    models['KNN'] = knn_model
    metrics_list.append({
        'Model': 'KNN',
        'MAE': mean_absolute_error(y_test, y_pred_knn),
        'MSE': mean_squared_error(y_test, y_pred_knn),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_knn)),
        'R2 Score': r2_score(y_test, y_pred_knn)
    })
    progress_bar.progress(100)
    status_text.text("✅ All models trained successfully!")
    
    # Display metrics comparison
    st.subheader("📈 Model Performance Comparison")
    metrics_df = pd.DataFrame(metrics_list)
    
    # Display metrics table
    st.dataframe(metrics_df.round(2))
    
    # Highlight best model
    best_model = metrics_df.loc[metrics_df['R2 Score'].idxmax(), 'Model']
    best_r2 = metrics_df['R2 Score'].max()
    st.success(f"🏆 **Best Model: {best_model}** with R² Score of {best_r2:.3f}")
    
    # Save best model
    best_model_obj = models[best_model]
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(best_model_obj, f)
    with open('label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    st.write("✅ Best model and label encoders saved")
    
    # Visualization
    st.header("📊 Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar plot for metrics
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        metrics_melted = metrics_df.melt(id_vars=['Model'], value_vars=['MAE', 'RMSE'], 
                                          var_name='Metric', value_name='Value')
        sns.barplot(data=metrics_melted, x='Model', y='Value', hue='Metric', ax=ax1)
        ax1.set_title('Error Metrics Comparison')
        ax1.tick_params(axis='x', rotation=45)
        st.pyplot(fig1)
    
    with col2:
        # R2 Score comparison
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.barplot(data=metrics_df, x='Model', y='R2 Score', ax=ax2, palette='viridis')
        ax2.set_title('R² Score Comparison')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        st.pyplot(fig2)
    
    # Feature importance for Random Forest
    if 'Random Forest' in models:
        st.subheader("🔍 Feature Importance (Random Forest)")
        rf_model = models['Random Forest']
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.barplot(data=feature_importance.head(10), x='Importance', y='Feature', ax=ax3, palette='viridis')
        ax3.set_title('Top 10 Feature Importances')
        st.pyplot(fig3)
    
    # Prediction section
    st.header("🎯 Make a Prediction")
    st.write("Use the trained models to predict salary based on input features")
    
    # Create input fields with appropriate types
    col1, col2 = st.columns(2)
    input_data = {}
    
    # Get list of feature names
    feature_names = X.columns.tolist()
    
    for i, feature in enumerate(feature_names):
        # Check if this feature was originally categorical
        if feature in categorical_cols:  # Now categorical_cols is a list, not an Index
            # Create dropdown with original category names
            with col1 if i % 2 == 0 else col2:
                selected_category = st.selectbox(
                    f"Select {feature}",
                    options=original_categories[feature],
                    key=f"select_{feature}"
                )
                # Encode the selected value
                input_data[feature] = label_encoders[feature].transform([selected_category])[0]
                # Show the encoded value in a small text (optional)
                st.caption(f"Encoded value: {input_data[feature]}")
        else:
            # Numeric field
            with col1 if i % 2 == 0 else col2:
                input_data[feature] = st.number_input(
                    f"Enter {feature}",
                    value=float(X[feature].mean()),
                    key=f"num_{feature}",
                    format="%.2f"
                )
    
    # Model selection for prediction
    selected_model_name = st.selectbox("Select Model for Prediction", list(models.keys()))
    selected_model = models[selected_model_name]
    
    if st.button("🔮 Predict Salary", type="primary"):
        try:
            input_df = pd.DataFrame([input_data])
            prediction = selected_model.predict(input_df)
            
            # Display prediction with nice formatting
            st.balloons()
            st.success(f"### 💰 Predicted Salary: **{prediction[0]:,.2f}**")
            
            # Show confidence based on R2 score
            model_r2 = metrics_df[metrics_df['Model'] == selected_model_name]['R2 Score'].values[0]
            confidence_level = "High" if model_r2 > 0.8 else "Medium" if model_r2 > 0.6 else "Low"
            st.info(f"📊 **Model Confidence:** {confidence_level} (R² Score: {model_r2:.2%})")
            
            # Show input summary
            with st.expander("View Input Summary"):
                st.write("**Your Input Values:**")
                for feature, value in input_data.items():
                    if feature in categorical_cols:
                        # Decode back to original value for display
                        original_value = label_encoders[feature].inverse_transform([int(value)])[0]
                        st.write(f"- {feature}: {original_value} (encoded: {value})")
                    else:
                        st.write(f"- {feature}: {value:,.2f}")
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

except FileNotFoundError:
    st.error("❌ Salary_Data.csv file not found! Please make sure the file is in the same directory.")
    import os
    st.write("Current working directory:", os.getcwd())
    st.write("Files in directory:", os.listdir('.'))

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.write("Please check your data file format and try again.")
