## Task 2: Data Preparation

### Description

This task focuses on preparing the dataset for use in a neural network model. The goal is to clean, transform, and organize the data to ensure it is suitable for training, validation, and testing.

### Steps

1. **Handle Missing Values**

   * Identify missing or null values in the dataset.
   * Apply appropriate techniques such as imputation (e.g., mean, median, mode) or removal, depending on the context of the feature.

2. **Encode Categorical Variables**

   * Convert categorical features into numerical representations compatible with neural networks.
   * Use suitable methods such as one-hot encoding or embeddings.

3. **Feature Scaling / Normalization**

   * Apply scaling techniques to numerical features to ensure consistent ranges.
   * Common methods include `StandardScaler` or `MinMaxScaler`.

4. **Feature Engineering (Optional)**

   * Create new features if they improve model performance or capture meaningful patterns in the data.

5. **Dataset Splitting**

   * Split the processed dataset into:

     * Training set (70%)
     * Validation set (15%)
     * Test set (15%)

### Output

* Once an Excel file path is provided, all preprocessing steps will be applied to the dataset.
* The processed datasets will be exported to the current working directory under:

  ```
  exports/
    └── <datetime>/
        ├── train_data.xlsx
        ├── val_data.xlsx
        └── test_data.xlsx
  ```
* File names and folder structure will clearly reflect their contents and creation time.

