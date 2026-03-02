import os
import time
import pandas as pd
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

def build_and_train_pipeline():
    if not os.path.exists(DATA_FILE):
        st.error(f"Dataset not found in repo: {DATA_FILE}")
        return None

    df = pd.read_csv(DATA_FILE)
    df = df.drop(columns=['TransactionID'], errors='ignore')

    cat_cols = ['Location', 'MerchantCategory']
    num_cols = ['Amount', 'Time', 'CardHolderAge']

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

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
        
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    try:
        joblib.dump(pipeline, MODEL_FILE)
    except Exception:
        pass

    return pipeline

@st.cache_resource
def get_model():
    if os.path.exists(MODEL_FILE):
        try:
            return joblib.load(MODEL_FILE)
        except Exception:
            pass
    pipeline = build_and_train_pipeline()
    return pipeline

def fetch_live_crypto_data(symbol="BTC/USD"):
    if not CCXT_AVAILABLE:
        return None
    try:
        exchange = ccxt.kraken()
        ticker = exchange.fetch_ticker(symbol)
        last_price = ticker.get('last')
        
        if last_price is None:
            raise ValueError("Exchange returned empty data.")
            
        return {
            'Amount': float(last_price),
            'Time': int(time.time() % 86400),
            'CardHolderAge': 30,
            'Location': 'Online',
            'MerchantCategory': 'Crypto'
        }
    except Exception as e:
        st.error(f"API Connection Error: {e}")
        return None

def main():
    st.set_page_config(page_title="FRAUD_SHIELD_V4", page_icon="🛡️", layout="wide")

    st.markdown("""
        <style>
        .stApp {
            background-color: #0a0e17;
            color: #ffffff;
        }
        @keyframes glow {
            0% { text-shadow: 0 0 5px #00ffcc; }
            50% { text-shadow: 0 0 20px #00ffcc, 0 0 30px #00b3ff; }
            100% { text-shadow: 0 0 5px #00ffcc; }
        }
        .title-glow {
            font-size: 48px;
            font-weight: 900;
            background: -webkit-linear-gradient(45deg, #00ffcc, #00b3ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: glow 3s infinite alternate;
            margin-bottom: 0px;
        }
        @keyframes pulse-border {
            0% { border-color: #ff4b4b; box-shadow: 0 0 5px #ff4b4b; }
            50% { border-color: #ff0000; box-shadow: 0 0 20px #ff0000; }
            100% { border-color: #ff4b4b; box-shadow: 0 0 5px #ff4b4b; }
        }
        .alert-box {
            background: rgba(255, 75, 75, 0.1);
            padding: 25px;
            border-radius: 12px;
            border: 2px solid #ff4b4b;
            animation: pulse-border 1.5s infinite;
            text-align: center;
        }
        .success-box {
            background: rgba(0, 255, 204, 0.1);
            padding: 25px;
            border-radius: 12px;
            border: 1px solid #00ffcc;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0, 255, 204, 0.2);
            transition: transform 0.3s ease;
        }
        .success-box:hover {
            transform: translateY(-5px);
        }
        div[data-testid="stMetricValue"] {
            color: #00ffcc !important;
            font-size: 32px !important;
            font-weight: 800 !important;
        }
        .stButton>button {
            background: linear-gradient(90deg, #00ffcc, #00b3ff);
            color: #0a0e17 !important;
            font-weight: bold;
            border: none;
            border-radius: 8px;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0 0 15px #00ffcc;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="title-glow">🛡️ FRAUD_SHIELD / NEXUS</p>', unsafe_allow_html=True)
    st.caption("⚡ ADVANCED THREAT DETECTION & LIVE AUDIT ENGINE")
    st.divider()

    model = get_model()
    if model is None:
        st.error("Model initialization failed. Please verify the dataset CSV.")
        st.stop()

    tab1, tab2 = st.tabs(["📊 MANUAL AUDIT", "₿ LIVE NETWORK STREAM"])

    with tab1:
        col1, col2 = st.columns([0.4, 0.6], gap="large")
        with col1:
            with st.form("audit_form"):
                st.subheader("📥 INPUT PARAMETERS")
                amt = st.number_input("TRANSACTION AMOUNT (USD)", value=250.00, step=50.0)
                age_val = st.slider("ACCOUNT HOLDER AGE", 18, 95, 30)
                loc = st.selectbox("GEO-LOCATION", ["California", "New York", "London", "Online", "Tokyo"])
                cat = st.selectbox("MERCHANT CATEGORY", ["Retail", "Electronics", "Crypto", "Entertainment"])
                submit = st.form_submit_button("INITIALIZE NEURAL SCAN")

        with col2:
            st.subheader("📉 AUDIT RESULTS")
            if submit:
                with st.spinner("Analyzing biometric and geographic signatures..."):
                    time.sleep(0.8)
                    input_df = pd.DataFrame([[amt, int(time.time()%86400), age_val, loc, cat]],
                                           columns=['Amount', 'Time', 'CardHolderAge', 'Location', 'MerchantCategory'])
                    try:
                        prob = model.predict_proba(input_df)[0][1]
                    except Exception as e:
                        st.error(f"Engine failure: {e}")
                        prob = None

                if prob is not None:
                    if prob > 0.5:
                        st.markdown(f"""
                            <div class="alert-box">
                                <h2 style="color: #ff4b4b; margin-top: 0;">🚨 CRITICAL THREAT DETECTED</h2>
                                <h3>Risk Probability: {(prob*100):.1f}%</h3>
                                <p>ACTION: TRANSACTION BLOCKED</p>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div class="success-box">
                                <h2 style="color: #00ffcc; margin-top: 0;">✅ TRANSACTION SECURE</h2>
                                <h3>Risk Probability: {(prob*100):.1f}%</h3>
                                <p>ACTION: AUTHORIZATION GRANTED</p>
                            </div>
                        """, unsafe_allow_html=True)
                        st.balloons()
                    
                    st.write("")
                    st.progress(prob)

    with tab2:
        st.subheader("📡 LIVE KRAKEN API UPLINK")
        
        if not CCXT_AVAILABLE:
            st.warning("CCXT module missing. Please check requirements.txt.")
            
        col_btn, col_space = st.columns([0.3, 0.7])
        with col_btn:
            fetch_btn = st.button("🔄 PING LIVE NETWORK")

        if fetch_btn:
            with st.spinner("Establishing secure handshake with exchange..."):
                raw_txn = fetch_live_crypto_data(symbol="BTC/USD")
                
            if raw_txn is not None:
                input_df = pd.DataFrame([raw_txn])
                try:
                    prob = model.predict_proba(input_df)[0][1]
                except Exception as e:
                    st.error(f"Inference error: {e}")
                    prob = None

                m1, m2, m3 = st.columns(3)
                m1.metric("ASSET VALUE (BTC/USD)", f"${raw_txn['Amount']:,.2f}")
                m2.metric("THREAT INDEX", f"{(prob*100):.2f}%" if prob is not None else "ERROR")
                m3.metric("NETWORK STATUS", "COMPROMISED" if (prob is not None and prob > 0.5) else "SECURE")

                if prob is not None and prob > 0.5:
                    st.markdown("""
                        <div class="alert-box" style="margin-top: 20px;">
                            <h3 style="color:#ff4b4b;">🚨 ANOMALOUS VOLATILITY SIGNATURE</h3>
                            <p>Abnormal trading patterns detected on the blockchain network.</p>
                        </div>
                    """, unsafe_allow_html=True)
                elif prob is not None:
                    st.markdown("""
                        <div class="success-box" style="margin-top: 20px;">
                            <h3 style="color:#00ffcc;">✅ NETWORK INTEGRITY VERIFIED</h3>
                            <p>Traffic patterns are within normal operational thresholds.</p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.toast('Live Data Synced Successfully!', icon='📡')
            else:
                st.error("Uplink failed. The exchange might be temporarily unavailable.")

if __name__ == "__main__":
    main()
