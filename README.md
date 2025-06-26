# Breast Cancer Classification - Real World ML Workflow

This project demonstrates a complete machine learning workflow for classifying breast cancer tumors as malignant or benign using the Breast Cancer Wisconsin dataset. The notebook guides you through data exploration, preprocessing, feature engineering, model selection, hyperparameter tuning, and evaluation.

## Project Overview

- **Dataset:** Breast Cancer Wisconsin (from `sklearn.datasets`)
- **Goal:** Predict whether a tumor is malignant or benign
- **Techniques:** Data cleaning, EDA, PCA, scaling, model training, cross-validation, and evaluation

## Workflow Steps

1. **Data Loading and Exploration**
    - Load and inspect the dataset structure, feature names, and class labels.
    - Check for missing values and class imbalance (dataset is slightly imbalanced, with more benign cases than malignant).

2. **Exploratory Data Analysis (EDA)**
    - Visualize feature correlations using a heatmap to identify multicollinearity.
    - Note strong correlations among certain feature groups.

3. **Feature Engineering**
    - Apply Principal Component Analysis (PCA) to highly correlated feature groups to reduce dimensionality.
    - Replace original correlated features with PCA components.

4. **Data Preprocessing**
    - Split data into training and testing sets with stratification.
    - Standardize features using `StandardScaler`.

5. **Reusable Functions**
    - Modular functions for model evaluation (confusion matrix, classification report, accuracy).
    - Modular function for hyperparameter tuning using `GridSearchCV`.

6. **Model Training and Evaluation**
    - Train and tune several models:
        - **K-Nearest Neighbors (KNN)**
        - **Decision Tree**
        - **Random Forest**
        - **Gaussian Naive Bayes**
        - **XGBoost**
    - Evaluate each model on the test set.

7. **Results and Comparison**
    - KNN and Random Forest achieved the highest accuracy (~96%).
    - Other models performed well but did not surpass KNN and RF.

8. **Conclusion**
    - Ensemble and instance-based methods (KNN, RF) are highly effective for this task.
    - Feature engineering (PCA, scaling) improved model performance.
    - Further improvements could include more extensive hyperparameter tuning or advanced ensemble techniques.

## Requirements

- Python 3.7+
- Jupyter Notebook
- numpy, pandas, matplotlib, seaborn, scikit-learn, xgboost

Install requirements with:
```sh
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

## Usage

Open the notebook in Jupyter and run the cells sequentially to reproduce the workflow and results.

## File Structure

- `index.ipynb` — Main notebook containing the entire workflow

## License

This project is for educational purposes.