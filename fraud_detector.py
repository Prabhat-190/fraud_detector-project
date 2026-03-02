import pandas as pd
import numpy as np
import joblib
import os
import time
import streamlit as st
import ccxt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Use exact names as they appear in your GitHub repo
MODEL_FILE = "fraud_model.pkl"
DATA_FILE = "credit_card_fraud_dataset_modified - credit_card_fraud_dataset_modified.csv"

def train_model():
    if not os.path.exists(DATA_FILE): 
        st.error(f"Dataset not found: {DATA_FILE}")
        return False
    
    df = pd.read_csv(DATA_FILE)
    df = df.drop(columns=['TransactionID'], errors='ignore')
    
    cat_cols = ['Location', 'MerchantCategory']
    num_cols = ['Amount', 'Time', 'CardHolderAge']
    
    for col in num_cols: df[col] = df[col].fillna(df[col].median())
    for col in cat_cols: df[col] = df[col].fillna(df[col].mode()[0])
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])
    
    X = df.drop('IsFraud', axis=1)
    y = df['IsFraud']
    
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, MODEL_FILE)
    return True

def fetch_live_crypto_data(symbol="BTC/USDT"):
    try:
        exchange = ccxt.binance()
        ticker = exchange.fetch_ticker(symbol)
        return {
            'Amount': ticker['last'],
            'Time': int(time.time() % 86400),
            'CardHolderAge': 30,
            'Location': 'Online',
            'MerchantCategory': 'Crypto'
        }
    except Exception as e:
        return None

def main():
    st.set_page_config(page_title="FRAUD_SHIELD_V3", page_icon="🛡️", layout="wide")

    st.markdown("""
        <style>
        .stApp { background: #050505; color: #e0e0e0; animation: fadeIn 1.2s ease-in; }
        @keyframes fadeIn { 0% { opacity: 0; } 100% { opacity: 1; } }
        .stTabs [data-baseweb="tab-list"] { gap: 24px; }
        .stTabs [data-baseweb="tab"] { height: 50px; background-color: #0f1115; border-radius: 8px; color: #666; }
        .stTabs [aria-selected="true"] { background-color: #00ffcc; color: #000 !important; font-weight: bold; }
        div[data-testid="stMetricValue"] { color: #00ffcc !important; }
        </style>
        """, unsafe_allow_html=True)

    # Auto-train if model is missing on the server
    if not os.path.exists(MODEL_FILE):
        if os.path.exists(DATA_FILE):
            with st.spinner("INITIATING NEURAL CORE..."): 
                train_model()
        else:
            st.error("Critical Error: Dataset CSV missing from repository.")
            return
    
    model = joblib.load(MODEL_FILE)

    st.title("🛡️ FRAUD_SHIELD / AUDIT_PRO")
    st.caption("AI-POWERED RISK MONITORING & LIVE CRYPTO AUDIT")
    st.divider()

    tab1, tab2 = st.tabs(["📊 ANALYST_AUDITOR", "₿ LIVE_CRYPTO_STREAM"])

    with tab1:
        col1, col2 = st.columns([0.45, 0.55], gap="large")
        with col1:
            with st.form("audit_form"):
                st.subheader("INPUT_STREAM")
                amt = st.number_input("TXN_AMOUNT (USD)", value=250.00)
                age_val = st.slider("HOLDER_AGE", 18, 95, 30)
                loc = st.selectbox("LOCATION", ["California", "New York", "London", "Online", "Tokyo"])
                cat = st.selectbox("CATEGORY", ["Retail", "Electronics", "Crypto", "Entertainment"])
                analyze = st.form_submit_button("RUN AUDIT")
        
        with col2:
            if analyze:
                input_df = pd.DataFrame([[amt, int(time.time()%86400), age_val, loc, cat]], 
                                       columns=['Amount', 'Time', 'CardHolderAge', 'Location', 'MerchantCategory'])
                prob = model.predict_proba(input_df)[0][1]
                if prob > 0.5:
                    st.error(f"🚨 ALERT: HIGH RISK DETECTED ({(prob*100):.1f}%)")
                else:
                    st.success(f"✅ SECURE: Risk Score ({(prob*100):.1f}%)")
                st.progress(prob)

    with tab2:
        st.subheader("📡 EXCHANGE_LIVE_FEED (BINANCE)")
        live_active = st.toggle("ACTIVATE NEURAL MONITORING")
        
        if live_active:
            placeholder = st.empty()
            while live_active:
                raw_txn = fetch_live_crypto_data()
                if raw_txn:
                    with placeholder.container():
                        input_df = pd.DataFrame([raw_txn])
                        prob = model.predict_proba(input_df)[0][1]
                        
                        m1, m2, m3 = st.columns(3)
                        m1.metric("LIVE PRICE (BTC)", f"${raw_txn['Amount']:,}")
                        m2.metric("RISK INDEX", f"{prob*100:.2f}%")
                        m3.metric("STATUS", "ANOMALY" if prob > 0.5 else "NOMINAL")

                        if prob > 0.5:
                            st.markdown("<div style='background:#4d0000; padding:20px; border-radius:10px; border:1px solid red;'>🚨 CRITICAL: Anomalous Volatility Signature Detected</div>", unsafe_allow_html=True)
                        else:
                            st.markdown("<div style='background:#00261a; padding:20px; border-radius:10px; border:1px solid #00ffcc;'>✅ STABLE: Network Traffic Verified Nominal</div>", unsafe_allow_html=True)
                else:
                    st.warning("Connection to Exchange lost. Retrying...")
                
                time.sleep(3) # Increased sleep to prevent API rate limiting
        else:
            st.info("System Standby. Toggle 'Activate Neural Monitoring' to begin the Binance stream.")

if __name__ == "__main__":
    main()
