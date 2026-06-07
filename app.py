import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import joblib

# Set Page Config
st.set_page_config(
    page_title="Telco Customer Churn Portal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS Styling
st.markdown("""
<style>
    /* Theme configuration */
    :root {
        --primary-color: #2b5c8f;
        --secondary-color: #f8f9fa;
        --accent-color: #d9534f;
    }
    
    /* Main body background and font styling */
    .main {
        background-color: #fafbfc;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f1f3f6;
    }
    
    /* Headers styling */
    h1, h2, h3 {
        color: #1a365d;
        font-weight: 700 !important;
    }
    
    /* Title Banner */
    .title-banner {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
    }
    .title-banner h1 {
        color: white !important;
        margin: 0;
        font-size: 2.5rem;
    }
    .title-banner p {
        margin-top: 0.5rem;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Card design */
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-top: 4px solid #1e3c72;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #1e3c72;
        margin-bottom: 0.2rem;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Custom tab headers */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
        font-size: 1rem;
        color: #4a5568;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #1e3c72;
    }
    .stTabs [aria-selected="true"] {
        color: #1e3c72 !important;
        border-bottom-color: #1e3c72 !important;
    }
    
    /* Risk Gauge Container */
    .risk-container {
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .risk-low {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .risk-medium {
        background: linear-gradient(135deg, #f857a6 0%, #ff5858 100%);
    }
    .risk-high {
        background: linear-gradient(135deg, #e53935 0%, #e35d5b 100%);
    }
</style>
""", unsafe_allow_html=True)

# Define columns matching training
NUMERICAL_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']
CATEGORICAL_COLS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]

# Helper function to load data science assets
@st.cache_resource
def load_ml_assets():
    try:
        preprocessor = joblib.load('models/preprocessor.joblib')
        models = {
            'Logistic Regression': joblib.load('models/logistic_regression.joblib'),
            'Random Forest': joblib.load('models/random_forest.joblib'),
            'XGBoost': joblib.load('models/xgboost.joblib')
        }
        with open('models/model_results.json', 'r') as f:
            metrics = json.load(f)
        return preprocessor, models, metrics
    except Exception as e:
        st.error(f"Error loading ML assets: {e}. Please run model training first (e.g. `python run.py --train`).")
        return None, None, None

# Load Preprocessor, Models, Metrics
preprocessor, models, metrics = load_ml_assets()

# Load Dataset for statistics
@st.cache_data
def get_raw_data():
    if os.path.exists('WA_Fn-UseC_-Telco-Customer-Churn.csv'):
        df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(r'^\s*$', '0.0', regex=True), errors='coerce').fillna(0.0)
        return df
    return None

df_raw = get_raw_data()

# App Header Banner
st.markdown("""
<div class="title-banner">
    <h1>Telco Churn Analytics Portal</h1>
    <p>Predict, Analyze, and Prevent Customer Attrition with Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab_overview, tab_model_perf, tab_predictor, tab_batch = st.tabs([
    "🏠 Churn Overview Dashboard",
    "📊 Model Performance Comparison",
    "🔮 Predict Churn (Single)",
    "📁 Batch Prediction (CSV)"
])

# ==========================================
# TAB 1: CHURN OVERVIEW DASHBOARD
# ==========================================
with tab_overview:
    st.header("Executive Summary")
    
    if df_raw is not None:
        total_cust = len(df_raw)
        churn_rate = (df_raw['Churn'].value_counts(normalize=True).get('Yes', 0) * 100)
        avg_tenure = df_raw['tenure'].mean()
        avg_charges = df_raw['MonthlyCharges'].mean()
        
        # Dashboard KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_cust:,}</div>
                <div class="metric-label">Total Customers</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card" style="border-top-color: #d9534f;">
                <div class="metric-value">{churn_rate:.1f}%</div>
                <div class="metric-label">Overall Churn Rate</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card" style="border-top-color: #2b5c8f;">
                <div class="metric-value">{avg_tenure:.1f} mo</div>
                <div class="metric-label">Average Tenure</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card" style="border-top-color: #f0ad4e;">
                <div class="metric-value">${avg_charges:.2f}</div>
                <div class="metric-label">Avg Monthly Charge</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br><hr><br>", unsafe_allow_html=True)
        
        # Key EDA Plots Display
        st.subheader("Key Insights from Data Science Analysis")
        col_plot1, col_plot2 = st.columns(2)
        
        with col_plot1:
            st.markdown("**Churn Distribution & Customer Demographics**")
            if os.path.exists('assets/churn_distribution.png'):
                st.image('assets/churn_distribution.png', use_container_width=True)
            else:
                st.info("Run `python run.py --eda` to pre-generate high-quality figures.")
                
            if os.path.exists('assets/tenure_distribution.png'):
                st.image('assets/tenure_distribution.png', use_container_width=True)
                
        with col_plot2:
            st.markdown("**Impact of Billing and Services**")
            if os.path.exists('assets/churn_by_contract.png'):
                st.image('assets/churn_by_contract.png', use_container_width=True)
                
            if os.path.exists('assets/churn_by_internet.png'):
                st.image('assets/churn_by_internet.png', use_container_width=True)
    else:
        st.warning("Dataset not found in workspace root. Please ensure 'WA_Fn-UseC_-Telco-Customer-Churn.csv' is available.")

# ==========================================
# TAB 2: MODEL PERFORMANCE COMPARISON
# ==========================================
with tab_model_perf:
    st.header("Machine Learning Model Evaluation")
    st.markdown("We trained and compared three distinct models. Below is the summarized comparison based on testing data.")
    
    if metrics is not None:
        metrics_df = pd.DataFrame(metrics).T
        
        col_m1, col_m2 = st.columns([1, 1.2])
        
        with col_m1:
            st.subheader("Metrics Summary Table")
            st.dataframe(metrics_df.style.highlight_max(axis=0, color='#d4edda'), use_container_width=True)
            
            st.markdown("""
            ### Understanding the Metrics:
            - **Accuracy**: The overall percentage of correct predictions.
            - **Precision**: Out of all predicted churners, how many actually churned? (Crucial to avoid offering discounts to loyal customers).
            - **Recall**: Out of all actual churners, how many did the model capture? (Crucial to ensure no high-risk customer goes unnoticed).
            - **F1-Score**: Harmonic mean of Precision and Recall.
            - **ROC-AUC**: Represents the model's ability to distinguish between churners and non-churners across all thresholds.
            """)
            
        with col_m2:
            st.subheader("ROC Curve Comparison")
            if os.path.exists('assets/roc_curve_comparison.png'):
                st.image('assets/roc_curve_comparison.png', use_container_width=True)
            else:
                st.info("ROC Curve image not found in assets.")
                
        st.markdown("<br><hr><br>", unsafe_allow_html=True)
        
        # Detail selection per model
        st.subheader("Detailed Model Diagnostics")
        selected_diag_model = st.selectbox("Select Model to View Detailed Diagnostics", list(models.keys()))
        model_key_name = selected_diag_model.lower().replace(" ", "_")
        
        col_diag1, col_diag2 = st.columns(2)
        with col_diag1:
            st.markdown(f"**Confusion Matrix - {selected_diag_model}**")
            cm_img_path = f'assets/confusion_matrix_{model_key_name}.png'
            if os.path.exists(cm_img_path):
                st.image(cm_img_path, use_container_width=True)
            else:
                st.info("Confusion Matrix plot not found.")
                
        with col_diag2:
            st.markdown(f"**Feature Importances/Coefficients - {selected_diag_model}**")
            feat_img_path = f'assets/feature_importance_{model_key_name}.png'
            if os.path.exists(feat_img_path):
                st.image(feat_img_path, use_container_width=True)
            else:
                st.info("Feature importance plot not found.")

# ==========================================
# TAB 3: PREDICT CHURN (SINGLE CUSTOMER)
# ==========================================
with tab_predictor:
    st.header("Real-Time Churn Risk Engine")
    st.markdown("Enter the customer's attributes below to compute their real-time probability of churn.")
    
    if models is not None and preprocessor is not None:
        # Layout input columns
        col_in1, col_in2, col_in3 = st.columns(3)
        
        with col_in1:
            st.subheader("Demographics")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Has Partner", ["Yes", "No"])
            dependents = st.selectbox("Has Dependents", ["Yes", "No"])
            
            st.subheader("Billing Details")
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", 
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            
        with col_in2:
            st.subheader("Services Subscriptions")
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            
            # Sub-services (only relevant if internet service exists)
            disabled_service = (internet == "No")
            security = st.selectbox("Online Security", ["No", "Yes", "No internet service"] if not disabled_service else ["No internet service"], disabled=disabled_service)
            backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"] if not disabled_service else ["No internet service"], disabled=disabled_service)
            protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"] if not disabled_service else ["No internet service"], disabled=disabled_service)
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"] if not disabled_service else ["No internet service"], disabled=disabled_service)
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"] if not disabled_service else ["No internet service"], disabled=disabled_service)
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"] if not disabled_service else ["No internet service"], disabled=disabled_service)

        with col_in3:
            st.subheader("Contract & Usage")
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12, step=1)
            monthly_charges = st.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0, step=0.5)
            
            # Pre-calculate Total Charges default
            default_total = float(tenure * monthly_charges)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=9000.0, value=default_total, step=50.0)
            
            st.subheader("Prediction Model")
            selected_model_name = st.selectbox("Select ML Model to Predict Churn", list(models.keys()))
            
        # Assemble dictionary
        cust_dict = {
            'gender': gender,
            'SeniorCitizen': senior,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone,
            'MultipleLines': multiple_lines,
            'InternetService': internet,
            'OnlineSecurity': security,
            'OnlineBackup': backup,
            'DeviceProtection': protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless,
            'PaymentMethod': payment,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        st.markdown("<br>", unsafe_allow_html=True)
        predict_button = st.button("🔮 Calculate Churn Risk Status", type="primary", use_container_width=True)
        
        if predict_button:
            model = models[selected_model_name]
            
            # Predict
            df_single = pd.DataFrame([cust_dict])
            processed = preprocessor.transform(df_single)
            prob = model.predict_proba(processed)[0, 1]
            prob_pct = prob * 100
            
            # Load utils logic directly to avoid import circular dependencies
            from src.utils import explain_prediction
            risk_factors, positive_factors = explain_prediction(cust_dict)
            
            # Visual presentation of risk level
            if prob_pct < 30:
                risk_class = "risk-low"
                risk_label = "LOW RISK"
                rec_actions = ["Maintain regular service communications.", "Check back for upgrade eligibility in 6 months."]
            elif prob_pct < 70:
                risk_class = "risk-medium"
                risk_label = "MEDIUM RISK"
                rec_actions = ["Target with personalized service discounts.", "Recommend updating to a 1-year contract to stabilize tenure."]
            else:
                risk_class = "risk-high"
                risk_label = "HIGH RISK"
                rec_actions = [
                    "Flag for immediate customer success outreach.",
                    "Offer retention discounts (e.g., $10-$15 off monthly charges).",
                    "Offer complimentary Tech Support or Online Security for 3 months."
                ]
                
            st.subheader("Risk Score Output")
            col_res1, col_res2 = st.columns([1, 1.5])
            
            with col_res1:
                st.markdown(f"""
                <div class="risk-container {risk_class}">
                    <div style="font-size: 1.1rem; font-weight: bold;">CHURN STATUS</div>
                    <div style="font-size: 2.8rem; font-weight: 800; margin: 0.5rem 0;">{prob_pct:.1f}%</div>
                    <div style="font-size: 1.2rem; font-weight: 800; letter-spacing: 1px;">{risk_label}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col_res2:
                st.markdown("### Suggested Retention Actions:")
                for action in rec_actions:
                    st.markdown(f"- **{action}**")
                    
            st.markdown("<br>", unsafe_allow_html=True)
            col_factors1, col_factors2 = st.columns(2)
            
            with col_factors1:
                st.markdown("### ⚠️ Risk Drivers (Increases Churn)")
                if len(risk_factors) > 0:
                    for factor in risk_factors:
                        st.error(factor)
                else:
                    st.write("No major risk factors detected.")
                    
            with col_factors2:
                st.markdown("### 🟢 Retention Anchors (Decreases Churn)")
                if len(positive_factors) > 0:
                    for factor in positive_factors:
                        st.success(factor)
                else:
                    st.write("No major positive anchors detected.")

# ==========================================
# TAB 4: BATCH PREDICTION (CSV UPLOAD)
# ==========================================
with tab_batch:
    st.header("Batch Customer Churn Predictions")
    st.markdown("Upload a CSV file containing multiple customer profiles (must contain similar headers to raw dataset) to run predictions on all of them simultaneously.")
    
    if models is not None and preprocessor is not None:
        selected_batch_model = st.selectbox("Select ML Model for Batch Inferences", list(models.keys()))
        
        # Download Sample Template
        st.markdown("### 1. Download Input Template")
        if df_raw is not None:
            # Create a sample sheet with first 5 rows without Churn column
            sample_df = df_raw.drop(columns=['Churn']).head(5)
            sample_csv = sample_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Sample CSV Template",
                data=sample_csv,
                file_name="telco_churn_batch_template.csv",
                mime="text/csv",
            )
            
        st.markdown("### 2. Upload Batch File")
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully!")
                
                # Check for needed columns
                missing_cols = [col for col in CATEGORICAL_COLS + ['tenure', 'MonthlyCharges', 'TotalCharges'] if col not in batch_df.columns]
                
                if len(missing_cols) > 0:
                    st.error(f"Missing required columns in CSV: {missing_cols}")
                else:
                    # Clean the dataset total charges just in case
                    batch_cleaned = batch_df.copy()
                    batch_cleaned['TotalCharges'] = batch_cleaned['TotalCharges'].replace(r'^\s*$', '0.0', regex=True)
                    batch_cleaned['TotalCharges'] = pd.to_numeric(batch_cleaned['TotalCharges'], errors='coerce').fillna(0.0)
                    batch_cleaned['SeniorCitizen'] = batch_cleaned['SeniorCitizen'].astype(str).str.replace('1', 'Yes').str.replace('0', 'No')
                    
                    # Preprocess
                    model = models[selected_batch_model]
                    processed_batch = preprocessor.transform(batch_cleaned)
                    
                    # Predict probabilities and classes
                    probs = model.predict_proba(processed_batch)[:, 1]
                    preds = model.predict(processed_batch)
                    
                    # Insert back into DF
                    batch_df['Churn_Probability'] = np.round(probs, 4)
                    batch_df['Churn_Prediction'] = np.where(preds == 1, 'Yes', 'No')
                    batch_df['Risk_Level'] = np.where(probs < 0.3, 'Low', np.where(probs < 0.7, 'Medium', 'High'))
                    
                    st.subheader("Batch Inference Results Preview")
                    st.dataframe(batch_df[['customerID', 'Contract', 'tenure', 'MonthlyCharges', 'Churn_Probability', 'Churn_Prediction', 'Risk_Level']].head(20), use_container_width=True)
                    
                    # Summary statistics
                    total_batch = len(batch_df)
                    pred_churners = (batch_df['Churn_Prediction'] == 'Yes').sum()
                    pred_churn_rate = (pred_churners / total_batch) * 100
                    
                    col_b1, col_b2, col_b3 = st.columns(3)
                    with col_b1:
                        st.metric("Total Batch Profiles", f"{total_batch:,}")
                    with col_b2:
                        st.metric("Predicted Churners", f"{pred_churners:,}")
                    with col_b3:
                        st.metric("Predicted Churn Rate", f"{pred_churn_rate:.1f}%")
                        
                    # Export button
                    results_csv = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Complete Predictions CSV",
                        data=results_csv,
                        file_name="telco_churn_batch_predictions.csv",
                        mime="text/csv",
                        type="primary"
                    )
            except Exception as e:
                st.error(f"Error processing upload: {e}")
