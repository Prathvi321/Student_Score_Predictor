# 🎓 Student Exam Score Predictor

An end-to-end Machine Learning project designed to predict a student's final math score based on their demographics, academic standing, and socio-economic factors. 

This repository encapsulates the full Data Science lifecycle: from Exploratory Data Analysis (EDA) and robust data preprocessing, to model training, evaluation, and finally deploying the predictive model into an interactive web dashboard.

## 🚀 Features

- **Exploratory Data Analysis (EDA)**: Comprehensive Jupyter Notebook breaking down statistical correlations, missing value mapping, and demographic distribution comparisons.
- **Robust Preprocessing Pipeline**: Automated handling of missing data, outlier normalization, and one-hot encoding (`drop_first=True`) to prevent multicollinearity traps.
- **Model Training & Comparison**: Scripted pipeline to quickly train, evaluate, and explicitly checkpoint multiple regression algorithms (Linear Regression & Decision Trees).
- **Interactive Web App**: A completely functioning, aesthetically designed `Streamlit` dashboard for zero-configuration, real-time user predictions.

## 🛠️ Tech Stack

- **Data Manipulation**: `pandas`, `NumPy`
- **Machine Learning**: `scikit-learn`
- **Visualization**: `Matplotlib`, `Seaborn`
- **Deployment Interface**: `Streamlit`
- **Serialization**: `joblib`

## 📁 Directory Structure

```text
Student_Score_Predictor/
├── app/                    # Streamlit dashboard interface
│   └── main.py
├── data/                   
│   ├── raw/                # Immutable original data 
│   └── processed/          # Cleaned, model-ready features
├── models/                 # Serialized .joblib model files
├── notebooks/              # Jupyter notebooks for EDA
│   └── EDA.ipynb
├── reports/figures/        # Generated metric plots (RMSE, R2, Feature Importance)
├── src/                    # Core pipeline codebase
│   ├── data_processing/    # Missing values & encoding logic
│   ├── evaluation/         # Model metrics & matplotlib charting
│   └── models/             # Training and serialization logic
└── requirements.txt        # Package dependencies
```

## ⚙️ Getting Started

### 1. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/Prathvi321/Student_Score_Predictor.git
cd "Student_Score_Predictor"
pip install -r requirements.txt
```

### 2. Run Exploratory Analysis
Examine the dataset correlations directly via the Jupyter instance:
```bash
jupyter notebook notebooks/EDA.ipynb
```

### 3. Training the Model

If you are setting this up for the first time, train the regression algorithms. This will populate the `models/` directory with serialized checkpoint engines.

```bash
python src/models/train.py
```

### 4. Launching the Web App

Start the interactive UI to test the dynamically generated models:

```bash
streamlit run app/main.py
```

---
*Developed for robust end-to-end data pipeline demonstration metrics under Python 3.*
