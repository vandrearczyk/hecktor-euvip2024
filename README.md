# Radiomics and Clinical Data Analysis with Machine Learning on HECKTOR data

## Overview
This notebook demonstrates how to load, preprocess, and analyze radiomics and clinical data using machine learning techniques. The goal is to classify patient outcomes using radiomics features (derived from imaging data and extracted using [QUANTIMAGE-V2](https://quantimage2.ehealth.hevs.ch/) and clinical features. The process includes data preprocessing, feature selection, model training, and evaluation. 
Data originates from 3D PET/CT images of the HEad and neCK TumOR segmentation and outcome prediction ([HECKTOR](https://hecktor.grand-challenge.org/)) challenge.  

## Steps in the Notebook

### 1. **Import Libraries**
   The notebook imports essential libraries for:
   - **Data handling**: `numpy`, `pandas`.
   - **Modeling and evaluation**: `scikit-learn` (for feature selection, train/test split, and model evaluation).
   - **Statistical analysis**: `scipy`, `statsmodels`.
   - **File handling**: `google.colab` (for uploading files in Colab).

### 2. **Load Data**
   - The following functions are provided to load and filter the data:
     - `load_features(folder_path, file_start)`: Loads CSV files from the specified folder and concatenates them.
     - `filter_patients(df1, df2)`: Filters patients present in both feature and outcome datasets.

### 3. **Preprocess Data**
   - **Data Cleaning and Transformation**:
     - `preprocess_data(df)`: Prepares and pivots the feature DataFrame to combine components like Modality and ROI.
     - `feature_preprocessing(df)`: Applies scaling, imputation, and one-hot encoding for categorical variables.
   - **Correlation Filtering**:
     - `drop_correlated_features(X_train, X_test, threshold)`: Drops highly correlated features to avoid multicollinearity.

### 4. **Feature Selection**
   - Manual selection of both radiomics and clinical features can be performed using `select_feature()` and the list `clinical_features`, respectively.
   - `SelectKBest` can be used to select the most important features based e.g. on mutual information between the features and target variable.

### 5. **Model Training and Evaluation**
   - **Model Selection**:
     - A `RandomForestClassifier` is used for this analysis, but the notebook is flexible to accommodate other models like `LogisticRegression`.
   - **Evaluation**:
     - Evaluation metrics include:
       - Accuracy
       - ROC AUC
       - Classification report
       - Confusion matrix
     - **Bootstrap Analysis**: Bootstrap resampling is used to compute confidence intervals for the ROC AUC score. This provides robust evaluation of model performance.
     - `evaluate_model(model, X_train, y_train, X_test, y_test, run_bootstrap)`: This function evaluates the model and outputs metrics.

## Key Functions

- **`load_features(folder_path, file_start)`**: Loads and concatenates CSV files containing patient features.
- **`preprocess_data(df)`**: Reshapes the feature DataFrame to combine components and prepare it for modeling.
- **`feature_preprocessing(df)`**: Performs scaling, imputation, and one-hot encoding of features.
- **`drop_correlated_features(X_train, X_test, threshold)`**: Removes highly correlated features to improve model performance.
- **`evaluate_model(model, X_train, y_train, X_test, y_test, run_bootstrap)`**: Evaluates the model on training and test sets, with an option to perform bootstrap analysis.

## Data Requirements
The notebook expects the following CSV files to be uploaded:
1. **Feature data**: CSV files containing radiomics features for each patient, each ROI (GTVp and GTVn) and each modality (PET and CT). Available in the `data/`
2. **Outcome data**: A CSV file containing patient outcomes for classification (e.g., `hecktor2022_HPV_outcomesBalanced.csv`, available in `data/`).
3. **Patient split**: A CSV file defining the training and test set split (e.g., `patient_split.csv`, available in `data/`).
4. **Clinical data**: A CSV file containing clinical information (e.g., `hecktor2022_clinicalFeatures.csv`, not publicly available).
