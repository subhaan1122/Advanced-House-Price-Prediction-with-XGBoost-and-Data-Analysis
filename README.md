🏡 House Price Prediction using XGBoost:

A machine learning project that predicts house sale prices using the Ames Housing Dataset. The project focuses on handling missing data, data preprocessing, feature engineering, and applying XGBoost Regression with GridSearchCV for hyperparameter tuning. It provides reliable price predictions for unseen test data.

📂 Project Overview:

This project:

- Loads training and test datasets from Google Drive
- Performs exploratory data analysis using visualizations
- Cleans and imputes missing data
- Applies one-hot encoding to categorical variables
- Builds a predictive model using XGBoost
- Evaluates model performance using Root Mean Squared Error (RMSE)
- Generates a submission CSV file for Kaggle or other evaluation

🚀 Features:

✅ Handles missing values (both numerical and categorical)
✅ Data cleaning and transformation
✅ Visualizes missing data via heatmaps
✅ Feature engineering using one-hot encoding
✅ XGBoost model with hyperparameter tuning (GridSearchCV)
✅ Submission-ready CSV output
✅ Evaluation using RMSE

📁 Dataset:

- Train: train.csv
- Test: test.csv
- Submission Template: sample_submission.csv

🔧 Libraries Used:

- pandas – data manipulation
- numpy – numerical operations
- matplotlib & seaborn – data visualization
- scikit-learn – train/test split, GridSearchCV, evaluation metrics
- xgboost – machine learning regression model

🧪 Model Training Steps:

1- Load Datasets:
Mount Google Drive and load training/test CSVs.

2- Visualize Missing Data
Use heatmaps to detect patterns in missing values.

3- Data Cleaning:

- Drop columns with too many missing values
- Impute numeric columns with mean
- Impute categorical columns with mode

4- Feature Engineering:

- Merge train and test data for consistent one-hot encoding
- Apply pd.get_dummies() to handle categorical data
- Ensure matching columns between train/test splits

5- Model Training:

- Use train_test_split() to create validation data
- Apply GridSearchCV to tune XGBRegressor parameters
- Evaluate model using RMSE

6- Prediction & Submission

- Predict on test set
- Save predictions as our_file.csv for submission

📊 Evaluation Metric:

RMSE (Root Mean Squared Error) is used to evaluate model accuracy on the validation set.
              " RMSE = sqrt(mean_squared_error(y_true, y_pred))"

▶️ How to Run:

1- Open the script in Google Colab.
2- Mount your Google Drive:              
                  " from google.colab import drive
                    drive.mount('/content/drive')"

3- Make sure the files are in your Drive 
4- Run all cells

📬 Contact:

Feel free to reach out if you want to collaborate or have suggestions:
- Name: Subhaan Khokhar
- Email: mskproductions2002@gmail.com
- LinkedIn: https://www.linkedin.com/in/muhammad-subhan-khokhar/


🏷️ Tags:
Python Machine Learning XGBoost Regression GridSearchCV Kaggle Ames Housing Feature Engineering Data Cleaning
