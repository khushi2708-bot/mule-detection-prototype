import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

st.set_page_config(page_title="CyberShield Prototype", layout="wide")

st.markdown("<h1 style='color:#00eaff;text-align:center;'>⚡ CyberShield: Mule Account Detection ⚡</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Large Dataset (CSV)", type=["csv"])

if uploaded_file:
    st.info("Processing large dataset in chunks...")
    chunks = pd.read_csv(uploaded_file, chunksize=10000)
    data = pd.concat(chunks)
    
    st.success(f"✅ Loaded {len(data)} records successfully!")

    # Feature Engineering
    data["txn_density"] = data["amount"] / (data["unique_senders"] + 1)
    
    # ML Model
    model = IsolationForest(contamination=0.05, random_state=42)
    data["risk_flag"] = model.fit_predict(data[["txn_density"]])
    data["risk_score"] = np.where(data["risk_flag"] == -1, np.random.randint(70,100), np.random.randint(0,50))

    st.subheader("⚠️ High‑Risk Accounts")
    high_risk = data[data["risk_score"] > 80]
    st.dataframe(high_risk.head())

    st.subheader("📈 Risk Score Distribution")
    st.bar_chart(data["risk_score"])
else:
    st.warning("Upload your dataset to start analysis.")
     
st.markdown("""
<style>
.stApp {
    background-color: #000000;
    color: #00eaff;
    font-family: monospace;
}
</style>
""", unsafe_allow_html=True)
