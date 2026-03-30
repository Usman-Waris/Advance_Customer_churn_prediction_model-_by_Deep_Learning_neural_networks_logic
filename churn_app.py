import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model
from thefuzz import process

# --- CONFIGURATION ---
st.set_page_config(page_title="Churn AI Predictor", layout="wide", page_icon="🧠")


# --- ASSETS LOADING ---
@st.cache_resource
def load_assets():
    try:
        # Paths ko apne system ke mutabiq check karlein
        model = load_model('churn_model_ann.h5')
        with open('scaler_ann.pkl', 'rb') as f:
            scaler = pickle.load(f)
        # Features list jo humne training mein use ki thi
        features = [
            'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
            'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
            'MultipleLines_No phone service', 'MultipleLines_Yes',
            'InternetService_Fiber optic', 'InternetService_No',
            'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
            'OnlineBackup_No internet service', 'OnlineBackup_Yes',
            'DeviceProtection_No internet service', 'DeviceProtection_Yes',
            'TechSupport_No internet service', 'TechSupport_Yes',
            'StreamingTV_No internet service', 'StreamingTV_Yes',
            'StreamingMovies_No internet service', 'StreamingMovies_Yes',
            'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
            'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
            'PaymentMethod_Mailed check'
        ]
        return model, scaler, features
    except Exception as e:
        st.error(f"Error loading model/scaler: {e}")
        return None, None, None


model, scaler, FEATURES = load_assets()


# --- SMART ALIGNMENT ENGINE ---
def align_data_smart(client_df, model_features):
    client_df.columns = [c.lower() for c in client_df.columns]

    industry_keys = {
        'MonthlyCharges': ['balance', 'monthly_spend', 'amount', 'charges', 'salary', 'spent'],
        'tenure': ['months', 'period', 'account_age', 'membership', 'days', 'duration'],
        'TotalCharges': ['total_balance', 'total_spent', 'cumulative', 'total_revenue'],
        'SeniorCitizen': ['age', 'is_old', 'senior', 'elderly'],
        'gender_Male': ['sex', 'is_male', 'gender', 'male']
    }

    mapped_df = pd.DataFrame(index=client_df.index)
    for target, alternatives in industry_keys.items():
        for alt in alternatives:
            if alt in client_df.columns:
                mapped_df[target] = client_df[alt]
                break

    client_cols = list(client_df.columns)
    for feature in model_features:
        if feature not in mapped_df.columns:
            match, score = process.extractOne(feature.lower(), client_cols)
            if score > 80:
                mapped_df[feature] = client_df[match]
            else:
                mapped_df[feature] = 0

    final_df = mapped_df.reindex(columns=model_features).fillna(0)
    return final_df.apply(pd.to_numeric, errors='coerce').fillna(0)


# --- UI DESIGN ---
st.title("🧠 Advanced Customer Churn Intelligence APP")
st.markdown("Deep Learning powered analysis for Bank,E-commerce data and each and every dataset related about churn.")

if model is None:
    st.warning("⚠️ Assets not found! Please ensure 'churn_model_ann.h5' and 'scaler_ann.pkl' are in the directory.")
else:
    tab1, tab2 = st.tabs(["📊 Batch Analysis", "🎯 Strategic Insights"])

    with tab1:
        uploaded_file = st.file_uploader("Upload Client Data (CSV)", type=["csv"])

        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            st.write("### Data Preview")
            st.dataframe(data.head(5))

            if st.button("Generate Intelligence Report"):
                with st.spinner("Analyzing patterns..."):
                    # 1. Align & Scale
                    ready_data = align_data_smart(data, FEATURES)
                    ready_data_scaled = scaler.transform(ready_data)

                    # 2. Predict
                    probs = model.predict(ready_data_scaled).flatten()

                    # 3. Advanced Dynamic Thresholding
                    low_t = np.percentile(probs, 70)
                    high_t = np.percentile(probs, 90)

                    conditions = [
                        (probs >= high_t),
                        (probs >= low_t) & (probs < high_t),
                        (probs < low_t)
                    ]
                    choices = ['🔥 Critical Risk', '⚠️ Medium Risk', '✅ Safe']

                    # 4. Results
                    data['Risk_Score'] = np.round(probs * 100, 2)
                    data['Status'] = np.select(conditions, choices, default='✅ Safe')

                    # KPI Summary
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Customers", len(data))
                    c2.metric("Critical Risk", len(data[data['Status'] == '🔥 Critical Risk']))
                    c3.metric("Avg Risk Score", f"{data['Risk_Score'].mean():.2f}%")

                    # Result Table
                    st.dataframe(
                        data[['Status', 'Risk_Score'] + [c for c in data.columns if c not in ['Status', 'Risk_Score']]])

                    # Download
                    csv = data.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Report 📥", csv, "Intelligence_Report.csv", "text/csv")

    with tab2:
        if 'data' in locals() and 'probs' in locals():
            st.subheader("Risk Distribution Analysis")
            st.bar_chart(data['Status'].value_counts())
            st.info(
                "The Intelligence Engine automatically adapted its thresholds based on the uploaded data distribution.")
        else:
            st.info("Please run a Batch Analysis first to see strategic insights.")

st.markdown("---")
st.caption("Developed by Usman Waris | Professional AI & Data Science")