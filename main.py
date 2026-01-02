import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from datetime import datetime

def prepare_and_export_data(file_path):
    # ---------------------------------------------------------
    # 1. Load Data
    # ---------------------------------------------------------
    print(f"Loading data from: {file_path}")
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # ---------------------------------------------------------
    # FIX: Drop columns that are completely empty (100% NaN)
    # This prevents the shape mismatch error when sklearn drops them internally.
    # ---------------------------------------------------------
    initial_shape = df.shape
    df = df.dropna(axis=1, how='all')
    print(f"Dropped {initial_shape[1] - df.shape[1]} empty columns.")
    print(f"New shape: {df.shape}")

    print(f"Original shape: {df.shape}") # This line can be removed or kept as is

    # ---------------------------------------------------------
    # 2. Initial Split (Train, Val, Test)
    # Split is done BEFORE preprocessing to prevent data leakage.
    # Ratios: 70% Train, 15% Val, 15% Test
    # ---------------------------------------------------------
    # First split: 85% Train+Val / 15% Test
    train_val_df, test_df = train_test_split(df, test_size=0.15, random_state=42)

    # Second split: 70% Train / 15% Val (from the remaining 85%)
    # Calculation: 0.15 / 0.85 ≈ 0.176
    train_df, val_df = train_test_split(train_val_df, test_size=0.176, random_state=42)

    print(f"Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # ---------------------------------------------------------
    # 3. Feature Engineering (Optional)
    # ---------------------------------------------------------
    # Example: Create an interaction feature between the first numeric and first categorical column found
    # This is a placeholder; modify this section based on your specific dataset domain knowledge.
    
    def apply_feature_engineering(data):
        data = data.copy()
        # Simple logic: Check if there are numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            # Example: Create a sum of the first two numeric columns
            col1, col2 = numeric_cols[0], numeric_cols[1]
            # Handling potential NaNs during creation just in case
            data[f'FE_Sum_{col1}_{col2}'] = data[col1].fillna(0) + data[col2].fillna(0)
        return data

    train_df = apply_feature_engineering(train_df)
    val_df = apply_feature_engineering(val_df)
    test_df = apply_feature_engineering(test_df)

    # ---------------------------------------------------------
    # 4. Separate Features and Target (Assumption)
    # Note: We assume the LAST column is the target. 
    # If your data has no target (unsupervised), remove the .iloc[:, :-1] parts.
    # ---------------------------------------------------------
    target_col_name = df.columns[-1]
    
    X_train = train_df.drop(columns=[target_col_name])
    y_train = train_df[target_col_name]
    
    X_val = val_df.drop(columns=[target_col_name])
    y_val = val_df[target_col_name]
    
    X_test = test_df.drop(columns=[target_col_name])
    y_test = test_df[target_col_name]

    # ---------------------------------------------------------
    # 5. Identify Column Types
    # ---------------------------------------------------------
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"Numeric features: {len(numeric_features)}, Categorical features: {len(categorical_features)}")

    # ---------------------------------------------------------
    # 6. Define Preprocessing Pipeline
    # ---------------------------------------------------------
    # Numeric: Impute with Median -> Scale with StandardScaler
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical: Impute with Mode (most_frequent) -> One-Hot Encode
    # handle_unknown='ignore' ensures val/test sets don't crash if they have new categories
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # ---------------------------------------------------------
    # 7. Fit and Transform
    # IMPORTANT: Fit ONLY on Training data
    # ---------------------------------------------------------
    print("Preprocessing data...")
    
    # Fit on training features
    preprocessor.fit(X_train)

    # Transform all sets
    X_train_processed = preprocessor.transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    # Get new column names after One-Hot Encoding
    # OneHotEncoder generates feature names automatically in newer sklearn versions
    try:
        ohe_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
        all_feature_names = numeric_features + list(ohe_feature_names)
    except AttributeError:
        # Fallback for older sklearn versions
        all_feature_names = [f"feature_{i}" for i in range(X_train_processed.shape[1])]

    # Convert back to DataFrame for easy export
    X_train_proc_df = pd.DataFrame(X_train_processed, columns=all_feature_names, index=X_train.index)
    X_val_proc_df = pd.DataFrame(X_val_processed, columns=all_feature_names, index=X_val.index)
    X_test_proc_df = pd.DataFrame(X_test_processed, columns=all_feature_names, index=X_test.index)

    # Combine Features and Target back together for export
    final_train = pd.concat([X_train_proc_df, y_train], axis=1)
    final_val = pd.concat([X_val_proc_df, y_val], axis=1)
    final_test = pd.concat([X_test_proc_df, y_test], axis=1)

    # ---------------------------------------------------------
    # 8. Export Data
    # ---------------------------------------------------------
    # Create folder structure: exports/datetime/
    base_dir = "exports"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    export_path = os.path.join(base_dir, timestamp)
    
    if not os.path.exists(export_path):
        os.makedirs(export_path)

    print(f"Exporting to: {export_path}")

    final_train.to_csv(os.path.join(export_path, "train_data.csv"), index=False)
    final_val.to_csv(os.path.join(export_path, "val_data.csv"), index=False)
    final_test.to_csv(os.path.join(export_path, "test_data.csv"), index=False)
    
    print("Task completed successfully.")

# --- Execution ---
# Please replace the path below with your actual Excel file path
excel_file_path = "crash_data_aa.xlsx" 

# Uncomment the line below to run:
prepare_and_export_data(excel_file_path)