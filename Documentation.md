# Data Preparation, Preprocessing, and Export Pipeline

## Overview

This script implements a **robust end-to-end data preprocessing pipeline** for machine learning workflows.
It loads data from an Excel file, performs cleaning, feature engineering, preprocessing, dataset splitting, and exports the processed datasets in a structured and reproducible way.

The pipeline is designed to:

* Prevent **data leakage**
* Handle **missing values**
* Support **numerical and categorical features**
* Be **scikit-learn compatible**
* Produce **ready-to-train datasets**

---

## Key Features

* 📂 Excel data ingestion
* 🧹 Automatic removal of empty columns
* ✂️ Train / Validation / Test splitting (70 / 15 / 15)
* 🧠 Optional feature engineering
* 🔢 Numerical preprocessing (imputation + scaling)
* 🏷️ Categorical preprocessing (imputation + one-hot encoding)
* 🚫 Safe handling of unseen categories
* 📤 Export of fully processed datasets with timestamps

---

## Dependencies

The script requires the following Python libraries:

```bash
pandas
numpy
scikit-learn
openpyxl
```

Make sure they are installed before running the script.

---

## Function: `prepare_and_export_data`

```python
prepare_and_export_data(file_path)
```

### Description

Processes an Excel dataset into clean, machine-learning-ready train, validation, and test CSV files.

### Parameters

| Parameter   | Type  | Description                            |
| ----------- | ----- | -------------------------------------- |
| `file_path` | `str` | Path to the input Excel (`.xlsx`) file |

### Returns

* None
* Outputs processed CSV files to disk

---

## Pipeline Breakdown

### 1. Data Loading

* Loads the dataset from an Excel file using `pandas.read_excel`
* Gracefully handles file loading errors

```python
df = pd.read_excel(file_path)
```

---

### 2. Empty Column Removal (Critical Fix)

Columns that are **100% missing values** are removed to prevent downstream shape mismatches during preprocessing.

```python
df = df.dropna(axis=1, how='all')
```

**Why this matters:**
Scikit-learn transformers silently drop empty columns, which can cause training/inference inconsistencies.

---

### 3. Dataset Splitting (Leakage-Safe)

Splitting is done **before preprocessing** to avoid data leakage.

| Dataset    | Percentage |
| ---------- | ---------- |
| Train      | 70%        |
| Validation | 15%        |
| Test       | 15%        |

```python
train_val_df, test_df = train_test_split(df, test_size=0.15)
train_df, val_df = train_test_split(train_val_df, test_size=0.176)
```

---

### 4. Feature Engineering (Optional)

A simple feature engineering function is applied consistently across all splits.

**Current implementation:**

* Creates a new feature that is the sum of the first two numeric columns (if available)

```python
data['FE_Sum_col1_col2'] = data[col1] + data[col2]
```

This section is modular and can be extended based on domain knowledge.

---

### 5. Feature–Target Separation

**Assumption:**
The **last column** of the dataset is the target variable.

```python
X = df.drop(columns=[target])
y = df[target]
```

> If your dataset does not follow this convention, adjust this logic accordingly.

---

### 6. Column Type Detection

Automatically identifies:

* **Numerical features** (`int`, `float`)
* **Categorical features** (`object`, `category`)

```python
numeric_features = X.select_dtypes(include=[np.number])
categorical_features = X.select_dtypes(include=['object'])
```

---

### 7. Preprocessing Pipelines

#### Numerical Pipeline

* Median imputation
* Standard scaling

```python
SimpleImputer(strategy='median')
StandardScaler()
```

#### Categorical Pipeline

* Most-frequent imputation
* One-hot encoding
* Ignores unseen categories during inference

```python
OneHotEncoder(handle_unknown='ignore')
```

#### Combined with `ColumnTransformer`

Ensures correct preprocessing per feature type.

---

### 8. Fitting and Transformation

* **Fit only on training data**
* Transform train, validation, and test sets using the same preprocessing rules

```python
preprocessor.fit(X_train)
preprocessor.transform(X_val)
preprocessor.transform(X_test)
```

---

### 9. Feature Name Recovery

* Retrieves feature names created by One-Hot Encoding
* Maintains interpretability in exported datasets

Fallback logic is included for older scikit-learn versions.

---

### 10. Exporting Processed Data

Processed datasets are exported as CSV files into a timestamped directory:

```text
exports/
└── 2026-01-02_14-32-10/
    ├── train_data.csv
    ├── val_data.csv
    └── test_data.csv
```

This ensures:

* Reproducibility
* Version control
* Experiment traceability

---

## How to Run

1. Update the Excel file path:

```python
excel_file_path = "crash_data_aa.xlsx"
```

2. Run the script:

```python
prepare_and_export_data(excel_file_path)
```

---

## Best Practices Followed

✔ No data leakage
✔ Consistent preprocessing
✔ Scikit-learn compliant
✔ Production-ready exports
✔ Extensible feature engineering

