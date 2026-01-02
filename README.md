## Environment Setup & Execution

Follow the steps below to set up a virtual environment, install dependencies, and run the data preprocessing pipeline.

### 1. Create a Virtual Environment

Create an isolated Python environment to avoid dependency conflicts:

```bash
python -m venv venv
```

---

### 2. Activate the Virtual Environment

**Windows (PowerShell / CMD):**

```bash
source venv/Scripts/activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

Once activated, your terminal prompt should indicate that you are inside the `venv` environment.

---

### 3. Install Required Dependencies

Ensure you have a `requirements.txt` file, then install all dependencies:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
pandas
numpy
scikit-learn
openpyxl
```

---

### 4. Run the Pipeline

Execute the main script:

```bash
python main.py
```

This will:

* Load the Excel dataset
* Preprocess and split the data
* Export the processed train, validation, and test datasets into the `exports/` directory

---

### 5. Output Location

Processed files will be saved under:

```text
exports/<timestamp>/
```

Example:

```text
exports/2026-01-02_14-32-10/
├── train_data.csv
├── val_data.csv
└── test_data.csv
```

---

## Notes

* Always activate the virtual environment before running the script
* Ensure the Excel file path in `main.py` is correct
* Each run creates a **new timestamped export folder** for reproducibility
