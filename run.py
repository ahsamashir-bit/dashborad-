import os
import sys
import argparse
import subprocess

# Add src to python path to allow imports when running run.py from root
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.eda_gen import generate_eda_plots
from src.data_processing import prepare_datasets
from src.model_training import train_and_evaluate_models

DATA_PATH = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'

def main():
    parser = argparse.ArgumentParser(description="Customer Churn Prediction Pipeline Runner")
    parser.add_argument('--eda', action='store_true', help="Run Exploratory Data Analysis & save plots")
    parser.add_argument('--train', action='store_true', help="Preprocess data & train ML models")
    parser.add_argument('--app', action='store_true', help="Launch Streamlit Dashboard")
    parser.add_argument('--all', action='store_true', help="Run all pipeline stages: EDA -> Train -> Launch App")
    
    args = parser.parse_args()
    
    # If no argument is provided, show help
    if not (args.eda or args.train or args.app or args.all):
        parser.print_help()
        sys.exit(1)
        
    if args.all:
        args.eda = True
        args.train = True
        args.app = True
        
    if args.eda:
        print("="*60)
        print("STAGE 1: Running Exploratory Data Analysis (EDA)...")
        print("="*60)
        generate_eda_plots(DATA_PATH)
        print("EDA generation complete.\n")
        
    if args.train:
        print("="*60)
        print("STAGE 2: Preprocessing Data & Training Machine Learning Models...")
        print("="*60)
        X_train, X_test, y_train, y_test, preprocessor = prepare_datasets(DATA_PATH)
        train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor)
        print("Model training and evaluation complete.\n")
        
    if args.app:
        print("="*60)
        print("STAGE 3: Launching Streamlit Dashboard App...")
        print("="*60)
        if not os.path.exists('models/preprocessor.joblib'):
            print("WARNING: Models and preprocessors not found. Running training first...")
            X_train, X_test, y_train, y_test, preprocessor = prepare_datasets(DATA_PATH)
            train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor)
            
        print("Starting Streamlit server... Press Ctrl+C to terminate.")
        try:
            # We run streamlit run app.py
            subprocess.run(["streamlit", "run", "app.py"], check=True)
        except KeyboardInterrupt:
            print("\nStreamlit server stopped.")
        except FileNotFoundError:
            print("ERROR: Streamlit is not installed or not in your PATH. Please install dependencies using 'pip install -r requirements.txt'")
            sys.exit(1)

if __name__ == '__main__':
    main()
