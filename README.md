# Customer Churn Prediction: End-to-End Machine Learning Solution

This repository contains a complete, professional, end-to-end data science and machine learning system built on the **Telco Customer Churn** dataset. The goal of this project is to analyze customer behaviors, predict churn probability with high precision and recall, evaluate and compare multiple machine learning algorithms, and expose these insights through a premium interactive web dashboard.

---

## 📂 Project Architecture

The project is structured modularly following software engineering best practices for data science:

```
project 3/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Raw customer data
│
├── notebooks/
│   └── eda.ipynb                              # Structured Jupyter Notebook for student EDA
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py                     # Data ingestion, cleaning & preprocessing pipelines
│   ├── model_training.py                      # Multi-model training, evaluation & metrics calculations
│   ├── eda_gen.py                             # Script to pre-generate high-quality plot assets
│   ├── create_notebook.py                     # Utility to construct/refresh the Jupyter notebook
│   └── utils.py                               # Helper modules for real-time predictions & business logic
│
├── models/
│   ├── preprocessor.joblib                    # Saved fitted preprocessing pipeline
│   ├── logistic_regression.joblib             # Saved trained Logistic Regression model
│   ├── random_forest.joblib                   # Saved trained Random Forest model
│   ├── xgboost.joblib                         # Saved trained XGBoost model
│   └── model_results.json                     # Performance metrics stored in JSON format
│
├── assets/                                    # Generated visual plots for the dashboard
│   ├── churn_distribution.png
│   ├── churn_by_contract.png
│   ├── tenure_distribution.png
│   ├── roc_curve_comparison.png
│   └── ...
│
├── app.py                                     # Interactive Streamlit Web Application
├── run.py                                     # Master execution entry point
├── requirements.txt                           # Pinpoint Python dependencies
└── README.md                                  # Detailed project documentation (this file)
```

---

## ⚙️ Installation & Setup

Ensure you have **Python 3.8+** installed. We recommend setting up a virtual environment.

1. **Clone or navigate to the workspace directory**:
   ```bash
   cd "workingfolder/project 3"
   ```

2. **Install requirements**:
   ```bash
   py -m pip install -r requirements.txt
   ```

---

## 🚀 Pipeline Execution

We have provided a master execution script `run.py` to manage the different stages of the machine learning pipeline.

### Stage 1: Run Exploratory Data Analysis (EDA)
Pre-generates high-quality charts showing customer distributions, contract styles, and service metrics.
```bash
py run.py --eda
```

### Stage 2: Train & Evaluate Models
Cleans the dataset, prepares training and testing sets, fits the preprocessing transformer, trains Logistic Regression, Random Forest, and XGBoost models, and generates evaluation plots.
```bash
py run.py --train
```

### Stage 3: Launch Streamlit Dashboard App
Launches the local Streamlit server for web interactions.
```bash
py run.py --app
```

### Run Full Pipeline
To run EDA, training, and start the app sequentially:
```bash
py run.py --all
```

---

## 📊 Performance Comparison & Results

After running `run.py --train`, models are evaluated on the test set (20% holdout). The results are saved in `models/model_results.json`:

| Model Name | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **80.6%** | 65.7% | **55.9%** | **60.4%** | **0.842** |
| **Random Forest** | 80.4% | **66.6%** | 52.7% | 58.8% | 0.838 |
| **XGBoost** | 79.7% | 64.5% | 52.4% | 57.8% | 0.841 |

> **Key Takeaway**: All three models perform strongly. Logistic Regression shows the best balance of recall and ROC-AUC. Random Forest yields slightly higher precision, which is useful when retention interventions are highly costly.

---

## 💻 Streamlit Web Application

The interactive web dashboard features four comprehensive sections:

1. **🏠 Churn Overview Dashboard**: Key business KPIs (overall churn rate, average monthly charges, total counts) and pre-generated data science plots exploring why customers leave.
2. **📊 Model Performance Comparison**: Compares the metrics table and displays the ROC Curve comparison. Allows selecting any model to view its specific confusion matrix and feature importances.
3. **🔮 Predict Churn (Single)**: An interactive form for real-time customer churn evaluation. Inputting demographics, contracts, and monthly billings outputs:
   - A color-coded Churn Risk Level (Low/Medium/High).
   - Real-time probability percentage.
   - Customized suggestion strategies.
   - Highlighted risk drivers and retention anchors.
4. **📁 Batch Prediction**: Allows uploading a CSV file of customer profiles, performing batch predictions, showing previews, and providing a download button to export the results.

---

## ☁️ Deployment Guide

Students can publish and share this project professionally:
1. **GitHub Repository**: Push this directory to your personal GitHub.
2. **Streamlit Community Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io).
   - Connect your GitHub repository.
   - Set the main file path to `app.py`.
   - Click "Deploy"! The app will be live and shareable.
