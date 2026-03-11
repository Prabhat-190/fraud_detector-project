import os
import time
import datetime
import random
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib

try:
    import ccxt
    CCXT_AVAILABLE = True
except Exception:
    CCXT_AVAILABLE = False

MODEL_FILE = "fraud_model_production_v1.pkl"

def generate_complex_dataset():
    np.random.seed(42)
    n = 25000
    
    amounts = np.random.lognormal(mean=4.0, sigma=2.0, size=n)
    times = np.random.randint(0, 86400, size=n)
    ages = np.random.randint(18, 80, size=n)
    locations = np.random.choice(["California", "New York", "London", "Online", "Tokyo", "Singapore"], size=n)
    cats = np.random.choice(["Retail", "Electronics", "Crypto", "Entertainment", "Travel"], size=n)
    
    risk_score = (amounts / 50000.0)
    risk_score += np.where((times < 18000) | (times > 79200), 0.15, 0.0)
    risk_score += np.where(ages < 25, 0.1, 0.0)
    risk_score += np.where((cats == "Crypto") & (locations == "Online"), 0.3, 0.0)
    risk_score += np.where(amounts > 15000, 0.4, 0.0)
    risk_score -= np.where(cats == "Retail", 0.1, 0.0)
    
    probs = 1 / (1 + np.exp(-(risk_score - 0.5) * 5))
    is_fraud = np.random.binomial(1, np.clip(probs, 0.01, 0.99))
    
    return pd.DataFrame({
        "Amount": amounts,
        "Time": times,
        "CardHolderAge": ages,
        "Location": locations,
        "MerchantCategory": cats,
        "IsFraud": is_fraud
    })

def build_production_model():
    df = generate_complex_dataset()
    cat_cols = ["Location", "MerchantCategory"]
    num_cols = ["Amount", "Time", "CardHolderAge"]
    
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])
    
    X = df.drop("IsFraud", axis=1)
    y = df["IsFraud"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=4, random_state=42, n_jobs=-1))
    ])
    
    pipeline.fit(X_train, y_train)
    try:
        joblib.dump(pipeline, MODEL_FILE)
    except Exception:
        pass
    return pipeline

@st.cache_resource
def load_system_model():
    if os.path.exists(MODEL_FILE):
        try:
            return joblib.load(MODEL_FILE)
        except Exception:
            pass
    return build_production_model()

def get_exchange_price(symbol="BTC/USD"):
    if not CCXT_AVAILABLE:
        return None
    try:
        exchange = ccxt.kraken()
        ticker = exchange.fetch_ticker(symbol)
        price = ticker.get("last")
        return float(price) if price else None
    except Exception:
        return None

def main():
    st.set_page_config(page_title="Fraud Shield | SOC", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
    
    if "ledger" not in st.session_state:
        st.session_state.ledger = pd.DataFrame(columns=["Timestamp", "Location", "Category", "Value (USD)", "Risk Score", "Status"])
    if "scan_count" not in st.session_state:
        st.session_state.scan_count = 0
        
    st.markdown("""
    <style>
    .stApp { background-color: #0b0f19; color: #e2e8f0; font-family: 'Inter', sans-serif; }
    .top-header { font-size: 38px; font-weight: 900; background: linear-gradient(90deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; padding-bottom: 0; margin-bottom: 0;}
    .sub-header { color: #94a3b8; font-size: 16px; font-weight: 500; margin-top: -5px; margin-bottom: 25px;}
    .soc-card { background: #1e293b; padding: 20px; border-radius: 12px; border: 1px solid #334155; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
    div[data-testid="stMetricValue"] { color: #38bdf8 !important; font-size: 32px !important; font-weight: 800 !important; }
    div[data-testid="stMetricLabel"] { color: #cbd5e1 !important; font-size: 14px !important; }
    .stButton>button { width: 100%; border-radius: 6px; background: #38bdf8; color: #0f172a; font-weight: 700; border: none; transition: 0.2s; }
    .stButton>button:hover { background: #7dd3fc; transform: translateY(-1px); }
    .risk-high { color: #ef4444; font-weight: bold; }
    .risk-low { color: #10b981; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)
    
    model = load_system_model()
    
    with st.sidebar:
        st.markdown("### ⚙️ SOC Controls")
        st.divider()
        block_threshold = st.slider("Auto-Block Threshold", 0.0, 1.0, 0.65, 0.01)
        sim_speed = st.slider("Live Polling Rate (s)", 1.0, 5.0, 2.0, 0.5)
        st.divider()
        st.markdown("#### System Metrics")
        st.metric("Model Architecture", "Random Forest")
        st.metric("Pipeline Stage", "Active / Deployed")
        st.metric("Total Scans", st.session_state.scan_count)
        if st.button("Purge System Ledger"):
            st.session_state.ledger = pd.DataFrame(columns=["Timestamp", "Location", "Category", "Value (USD)", "Risk Score", "Status"])
            st.session_state.scan_count = 0
            st.rerun()

    st.markdown('<div class="top-header">🛡️ FRAUD SHIELD SOC</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">SECURITY OPERATIONS CENTER · REAL-TIME INFERENCE ENGINE</div>', unsafe_allow_html=True)

    tab_live, tab_manual = st.tabs(["📡 LIVE NETWORK TELEMETRY", "🔬 MANUAL INVESTIGATION"])

    with tab_manual:
        c1, c2 = st.columns([1, 1.5], gap="large")
        with c1:
            st.markdown('<div class="soc-card">', unsafe_allow_html=True)
            st.subheader("Input Parameters")
            with st.form("manual_scan"):
                amt = st.number_input("Transaction Value (USD)", min_value=1.0, value=1500.0, step=100.0)
                age = st.slider("Subject Age", 18, 90, 28)
                loc = st.selectbox("Origin Node", ["California", "New York", "London", "Online", "Tokyo", "Singapore"])
                cat = st.selectbox("Transaction Class", ["Retail", "Electronics", "Crypto", "Entertainment", "Travel"])
                run_scan = st.form_submit_button("EXECUTE NEURAL SCAN")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with c2:
            st.markdown('<div class="soc-card" style="min-height: 400px;">', unsafe_allow_html=True)
            st.subheader("Inference Results")
            if run_scan:
                st.session_state.scan_count += 1
                df_eval = pd.DataFrame([[amt, int(time.time() % 86400), age, loc, cat]], columns=["Amount", "Time", "CardHolderAge", "Location", "MerchantCategory"])
                
                start_time = time.perf_counter()
                prob = model.predict_proba(df_eval)[0][1]
                latency = (time.perf_counter() - start_time) * 1000
                
                st.progress(prob)
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Risk Probability", f"{prob*100:.2f}%")
                m2.metric("Inference Latency", f"{latency:.2f} ms")
                m3.metric("System Decision", "BLOCK" if prob >= block_threshold else "ALLOW")
                
                if prob >= block_threshold:
                    st.error(f"🚨 SIGNATURE MATCH: High probability of unauthorized activity. Transaction terminated.")
                else:
                    st.success(f"✅ VERIFIED: Telemetry falls within normal operational parameters.")
            else:
                st.info("Awaiting parameter input. Click 'Execute Neural Scan' to begin.")
            st.markdown('</div>', unsafe_allow_html=True)

    with tab_live:
        st.markdown('<div class="soc-card">', unsafe_allow_html=True)
        col_ctrl, col_status = st.columns([0.3, 0.7])
        with col_ctrl:
            is_streaming = st.toggle("🟢 ENGAGE LIVE UPLINK")
        
        live_ui = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

        if is_streaming:
            while is_streaming:
                with live_ui.container():
                    price = get_exchange_price("BTC/USD")
                    if not price:
                        price = 65000.0 + random.uniform(-100, 100)
                        
                    current_ts = datetime.datetime.now().strftime("%H:%M:%S")
                    rand_loc = random.choice(["California", "New York", "London", "Online", "Tokyo", "Singapore"])
                    rand_cat = random.choice(["Crypto", "Electronics", "Travel"])
                    rand_age = random.randint(18, 70)
                    
                    df_live = pd.DataFrame([[price, int(time.time() % 86400), rand_age, rand_loc, rand_cat]], columns=["Amount", "Time", "CardHolderAge", "Location", "MerchantCategory"])
                    risk_val = model.predict_proba(df_live)[0][1]
                    st.session_state.scan_count += 1
                    
                    status_str = "BLOCKED" if risk_val >= block_threshold else "CLEARED"
                    
                    new_log = pd.DataFrame([{
                        "Timestamp": current_ts,
                        "Location": rand_loc,
                        "Category": rand_cat,
                        "Value (USD)": f"${price:,.2f}",
                        "Risk Score": risk_val,
                        "Status": status_str
                    }])
                    
                    st.session_state.ledger = pd.concat([new_log, st.session_state.ledger]).head(8)
                    
                    met1, met2, met3 = st.columns(3)
                    met1.metric("Latest Asset Value", f"${price:,.2f}")
                    met2.metric("Live Threat Index", f"{risk_val*100:.2f}%")
                    met3.metric("Network Status", "ANOMALY DETECTED" if risk_val >= block_threshold else "NOMINAL")
                    
                    st.markdown("### Transaction Ledger")
                    
                    styled_ledger = st.session_state.ledger.style.map(
                        lambda v: 'color: #ef4444; font-weight: bold;' if v == 'BLOCKED' else ('color: #10b981; font-weight: bold;' if v == 'CLEARED' else ''),
                        subset=['Status']
                    ).format({"Risk Score": "{:.2%}"})
                    
                    st.dataframe(styled_ledger, use_container_width=True, hide_index=True)
                    
                time.sleep(sim_speed)
        else:
            with live_ui.container():
                st.info("Live network uplink disconnected. Toggle switch to resume data ingestion.")
                if not st.session_state.ledger.empty:
                    st.markdown("### Previous Ledger Data")
                    st.dataframe(st.session_state.ledger, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
