import os
import time
import datetime
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

MODEL_FILE = "fraud_model.pkl"
DATA_FILE = "credit_card_fraud_dataset_modified - credit_card_fraud_dataset_modified.csv"

# ==========================================
# 1. CORE MACHINE LEARNING PIPELINE
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
    
    # Tuned for production variance
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42, k_neighbors=2)), 
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42))
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
# 2. REAL-TIME EXCHANGE API (TRADE LEVEL)
# ==========================================
def fetch_live_trade(symbol="BTC/USD"):
    """Fetches the actual last executed transaction on the exchange, not just the price."""
    if not CCXT_AVAILABLE: return None
    try:
        exchange = ccxt.kraken()
        # Fetch the single most recent trade executed on the network
        trades = exchange.fetch_trades(symbol, limit=1)
        if not trades: return None
        
        last_trade = trades[0]
        trade_price = last_trade['price']
        trade_amount_crypto = last_trade['amount']
        trade_usd_value = trade_price * trade_amount_crypto
        
        return {
            'Amount': float(trade_usd_value),
            'Time': int(time.time() % 86400),
            'CardHolderAge': 30, # Defaulted for crypto anonymity
            'Location': 'Online',
            'MerchantCategory': 'Crypto',
            'Side': last_trade['side'].upper(), # Buy or Sell
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

    # Initialize memory for live terminal
    if 'trade_history' not in st.session_state:
        st.session_state.trade_history = pd.DataFrame(columns=['Timestamp', 'Type', 'Crypto Amount', 'USD Value', 'Risk Score'])

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
        }
        div[data-testid="stMetricValue"] { color: #00ffcc !important; font-size: 28px !important; }
        .risk-high { color: #ff4b4b; font-weight: bold; }
        .risk-low { color: #00ffcc; font-weight: bold; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="title-glow">🛡️ FRAUD_SHIELD / ENTERPRISE</p>', unsafe_allow_html=True)
    st.caption("⚡ REAL-TIME TRANSACTION LEDGER AUDITOR")
    st.divider()

    model = get_model()
    if model is None:
        st.error("SYSTEM HALTED: Dataset missing.")
        st.stop()

    tab1, tab2 = st.tabs(["📊 MANUAL INTERROGATION", "📡 LIVE NETWORK AUDIT (REAL-TIME)"])

    # --- TAB 1: MANUAL (Kept intact) ---
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
        
        # UI Placeholders for the live loop
        metrics_placeholder = st.empty()
        terminal_placeholder = st.empty()

        if live_active:
            while live_active:
                trade = fetch_live_trade(symbol="BTC/USD")
                
                if trade is not None:
                    # ML Inference on the EXACT transaction size
                    input_df = pd.DataFrame([trade])
                    prob = model.predict_proba(input_df[['Amount', 'Time', 'CardHolderAge', 'Location', 'MerchantCategory']])[0][1]
                    
                    # Log data to memory for the live terminal
                    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    new_row = pd.DataFrame({
                        'Timestamp': [timestamp],
                        'Type': [trade['Side']],
                        'Crypto Amount': [f"{trade['CryptoAmount']:.4f} BTC"],
                        'USD Value': [f"${trade['Amount']:,.2f}"],
                        'Risk Score': [prob]
                    })
                    st.session_state.trade_history = pd.concat([new_row, st.session_state.trade_history]).head(10) # Keep last 10
                    
                    # Update Metrics UI
                    with metrics_placeholder.container():
                        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("LATEST TRADE (USD)", f"${trade['Amount']:,.2f}")
                        m2.metric("ORDER TYPE", trade['Side'])
                        m3.metric("BTC PRICE", f"${trade['Price']:,.2f}")
                        
                        # Dynamic color for risk metric
                        risk_pct = prob * 100
                        m4.metric("NEURAL RISK", f"{risk_pct:.1f}%")
                        st.markdown("</div>", unsafe_allow_html=True)

                        if prob > 0.5:
                            st.error(f"🚨 FRAUD ALERT: High-risk anomaly detected on transaction worth ${trade['Amount']:,.2f}")

                    # Update Terminal UI
                    with terminal_placeholder.container():
                        st.markdown("### 💻 LIVE AUDIT TERMINAL")
                        # Format the risk score for display
                        display_df = st.session_state.trade_history.copy()
                        display_df['Risk Score'] = display_df['Risk Score'].apply(
                            lambda x: f"🔴 {x*100:.1f}%" if x > 0.5 else f"🟢 {x*100:.1f}%"
                        )
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                time.sleep(1.5) # Poll the exchange ledger every 1.5 seconds
        else:
            st.info("System Standby. Activate the toggle to begin monitoring real-time exchange traffic.")

if __name__ == "__main__":
    main()
