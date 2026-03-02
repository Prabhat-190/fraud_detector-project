import os
import time
import datetime
import random
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

try:
    import ccxt
    CCXT_AVAILABLE = True
except Exception:
    CCXT_AVAILABLE = False

# Renamed to force the server to train the new 500-Tree model automatically
MODEL_FILE = "fraud_model_v6.pkl" 
DATA_FILE = "credit_card_fraud_dataset_modified - credit_card_fraud_dataset_modified.csv"

# ==========================================
# 1. HIGH-PRECISION MACHINE LEARNING PIPELINE
# ==========================================
def build_and_train_pipeline():
    if not os.path.exists(DATA_FILE): return None
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
        ('smote', SMOTE(random_state=42, k_neighbors=2)), 
        # UPGRADED: 500 trees and deeper logic for ultra-precise decimal probabilities
        ('classifier', RandomForestClassifier(n_estimators=500, max_depth=12, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    try: joblib.dump(pipeline, MODEL_FILE)
    except Exception: pass
    return pipeline

@st.cache_resource
def get_model():
    if os.path.exists(MODEL_FILE):
        try: return joblib.load(MODEL_FILE)
        except Exception: pass
    return build_and_train_pipeline()

# ==========================================
# 2. DYNAMIC REAL-TIME EXCHANGE API
# ==========================================
def fetch_live_trade(symbol="BTC/USD"):
    """Fetches the actual last executed transaction and simulates global user traffic."""
    if not CCXT_AVAILABLE: return None
    try:
        exchange = ccxt.kraken()
        trades = exchange.fetch_trades(symbol, limit=1)
        if not trades: return None
        
        last_trade = trades[0]
        trade_price = last_trade['price']
        trade_amount_crypto = last_trade['amount']
        trade_usd_value = trade_price * trade_amount_crypto
        
        # DYNAMIC PROFILING: Mimics different users hitting the exchange in real-time
        locations = ["California", "New York", "London", "Online", "Tokyo", "Berlin", "Paris"]
        
        return {
            'Amount': float(trade_usd_value),
            'Time': int(time.time() % 86400),
            'CardHolderAge': random.randint(18, 75), # Dynamic Age
            'Location': random.choice(locations),    # Dynamic Geo-Node
            'MerchantCategory': 'Crypto',
            'Side': last_trade['side'].upper(),
            'CryptoAmount': trade_amount_crypto,
            'Price': trade_price
        }
    except Exception as e:
        return None

# ==========================================
# 3. PRODUCTION UI & STREAMING DASHBOARD
# ==========================================
def main():
    st.set_page_config(page_title="FRAUD_SHIELD_PRO", page_icon="🛡️", layout="wide")

    if 'trade_history' not in st.session_state:
        st.session_state.trade_history = pd.DataFrame(columns=['Timestamp', 'Type', 'Crypto Amount', 'USD Value', 'Location', 'Risk Score'])

    st.markdown("""
        <style>
        .stApp { background-color: #060a11; color: #ffffff; }
        .title-glow {
            font-size: 42px; font-weight: 900;
            background: linear-gradient(90deg, #00ffcc, #00b3ff);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 0px;
        }
        .metric-container {
            background: #0f1522; padding: 20px; border-radius: 10px; border: 1px solid #1e293b;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
        }
        div[data-testid="stMetricValue"] { color: #00ffcc !important; font-size: 28px !important; }
        .stProgress > div > div > div > div {
            background-image: linear-gradient(to right, #00ffcc, #00b3ff);
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="title-glow">🛡️ FRAUD_SHIELD / ENTERPRISE</p>', unsafe_allow_html=True)
    st.caption("⚡ HIGH-PRECISION TRANSACTION LEDGER AUDITOR")
    st.divider()

    model = get_model()
    if model is None:
        st.error("SYSTEM HALTED: Dataset missing.")
        st.stop()

    tab1, tab2 = st.tabs(["📊 MANUAL INTERROGATION", "📡 LIVE NETWORK AUDIT (REAL-TIME)"])

    # --- TAB 1: MANUAL AUDIT ---
    with tab1:
        col1, col2 = st.columns([0.4, 0.6], gap="large")
        with col1:
            with st.form("audit_form"):
                st.subheader("📥 TRANSACTION PARAMETERS")
                amt = st.number_input("USD AMOUNT", value=250.00, step=50.0)
                age_val = st.slider("HOLDER AGE", 18, 95, 30)
                loc = st.selectbox("LOCATION", ["California", "New York", "London", "Online", "Tokyo"])
                cat = st.selectbox("CATEGORY", ["Retail", "Electronics", "Crypto", "Entertainment"])
                submit = st.form_submit_button("RUN NEURAL SCAN")

        with col2:
            if submit:
                input_df = pd.DataFrame([[amt, int(time.time()%86400), age_val, loc, cat]],
                                       columns=['Amount', 'Time', 'CardHolderAge', 'Location', 'MerchantCategory'])
                prob = model.predict_proba(input_df)[0][1]
                
                st.progress(prob)
                if prob > 0.5:
                    st.error(f"🚨 CRITICAL RISK: {(prob*100):.2f}% - TRANSACTION BLOCKED")
                else:
                    st.success(f"✅ VERIFIED: {(prob*100):.2f}% - TRANSACTION SECURE")

    # --- TAB 2: TRUE REAL-TIME PRODUCTION STREAM ---
    with tab2:
        st.subheader("📡 KRAKEN LIVE LEDGER (BTC/USD)")
        
        col_toggle, col_status = st.columns([0.3, 0.7])
        with col_toggle:
            live_active = st.toggle("🟢 ACTIVATE LIVE AUDIT", value=False)
        
        metrics_placeholder = st.empty()
        terminal_placeholder = st.empty()

        if live_active:
            while live_active:
                trade = fetch_live_trade(symbol="BTC/USD")
                
                if trade is not None:
                    # ML Inference on the EXACT transaction
                    input_df = pd.DataFrame([trade])
                    prob = model.predict_proba(input_df[['Amount', 'Time', 'CardHolderAge', 'Location', 'MerchantCategory']])[0][1]
                    
                    # Log data to memory for the live terminal
                    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    new_row = pd.DataFrame({
                        'Timestamp': [timestamp],
                        'Type': [trade['Side']],
                        'Crypto Amount': [f"{trade['CryptoAmount']:.4f} BTC"],
                        'USD Value': [f"${trade['Amount']:,.2f}"],
                        'Location': [f"Node: {trade['Location']}"],
                        'Risk Score': [prob]
                    })
                    st.session_state.trade_history = pd.concat([new_row, st.session_state.trade_history]).head(10)
                    
                    # Update Metrics UI
                    with metrics_placeholder.container():
                        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("LATEST TRADE (USD)", f"${trade['Amount']:,.2f}")
                        m2.metric("ORDER TYPE", trade['Side'])
                        m3.metric("BTC PRICE", f"${trade['Price']:,.2f}")
                        
                        risk_pct = prob * 100
                        m4.metric("NEURAL RISK", f"{risk_pct:.2f}%")
                        st.markdown("</div>", unsafe_allow_html=True)

                        if prob > 0.5:
                            st.error(f"🚨 FRAUD ALERT: High-risk anomaly detected from {trade['Location']} node.")

                    # Update Terminal UI
                    with terminal_placeholder.container():
                        st.markdown("### 💻 LIVE AUDIT TERMINAL")
                        display_df = st.session_state.trade_history.copy()
                        display_df['Risk Score'] = display_df['Risk Score'].apply(
                            lambda x: f"🔴 {x*100:.2f}%" if x > 0.5 else f"🟢 {x*100:.2f}%"
                        )
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                time.sleep(1.5) 
        else:
            st.info("System Standby. Activate the toggle to begin monitoring real-time exchange traffic.")

if __name__ == "__main__":
    main()
