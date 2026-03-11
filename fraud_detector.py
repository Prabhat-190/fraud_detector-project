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
except:
    CCXT_AVAILABLE = False

MODEL_FILE = "fraud_model_v17.pkl"

def generate_continuous_data():
    np.random.seed(42)
    n = 15000
    amounts = np.random.uniform(10.0, 100000.0, size=n)
    times = np.random.randint(0, 86400, size=n)
    ages = np.random.randint(18, 80, size=n)
    locations = np.random.choice(["California", "New York", "London", "Online", "Tokyo", "Singapore"], size=n)
    cats = np.random.choice(["Retail", "Electronics", "Crypto", "Entertainment", "Travel"], size=n)
    
    risk_score = (amounts / 60000.0)
    risk_score += np.where((times < 18000) | (times > 79200), 0.15, 0.0)
    risk_score += np.where(ages < 30, 0.1, 0.0)
    risk_score += np.where((cats == "Crypto") & (locations == "Online"), 0.25, 0.0)
    risk_score += np.where(amounts > 20000, 0.3, 0.0)
    risk_score -= np.where(cats == "Retail", 0.1, 0.0)
    
    probs = 1 / (1 + np.exp(-(risk_score - 0.5) * 4))
    is_fraud = np.random.binomial(1, np.clip(probs, 0.01, 0.99))
    
    return pd.DataFrame({"Amount": amounts, "Time": times, "CardHolderAge": ages, "Location": locations, "MerchantCategory": cats, "IsFraud": is_fraud})

def build_fluid_model():
    df = generate_continuous_data()
    cat_cols = ["Location", "MerchantCategory"]
    num_cols = ["Amount", "Time", "CardHolderAge"]
    preprocessor = ColumnTransformer([("num", StandardScaler(), num_cols), ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])
    X = df.drop("IsFraud", axis=1)
    y = df["IsFraud"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor), 
        ("classifier", RandomForestClassifier(n_estimators=250, max_depth=None, min_samples_leaf=2, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    try:
        joblib.dump(pipeline, MODEL_FILE)
    except:
        pass
    return pipeline

@st.cache_resource
def get_model():
    if os.path.exists(MODEL_FILE):
        try:
            return joblib.load(MODEL_FILE)
        except:
            pass
    return build_fluid_model()

def get_live_price():
    if not CCXT_AVAILABLE:
        return None
    try:
        exchange = ccxt.kraken()
        ticker = exchange.fetch_ticker("BTC/USD")
        return float(ticker.get("last")) if ticker.get("last") else None
    except:
        return None

def main():
    st.set_page_config(page_title="Fraud Shield Enterprise", layout="wide")
    
    if "manual_history" not in st.session_state:
        st.session_state.manual_history = []
    if "live_history" not in st.session_state:
        st.session_state.live_history = []
        
    st.markdown("""
    <style>
    .stApp { background: linear-gradient(180deg, #051017 0%, #0a1922 100%); color: #ffffff; }
    
    .header-container { display: flex; align-items: center; margin-bottom: 0px; }
    .header-icon { font-size: 38px; color: #00ffd0; margin-right: 12px; }
    .header-text { font-size: 42px; font-weight: 900; color: #00ffd0; margin: 0; letter-spacing: 1px; }
    .sub-text { color: #829ab1; font-size: 14px; margin-top: 5px; margin-bottom: 30px; }
    
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p { font-size: 15px; font-weight: 600; color: #829ab1; }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] { border-bottom-color: #ef4444 !important; }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] [data-testid="stMarkdownContainer"] p { color: #ef4444 !important; }
    
    .input-card { background-color: #101c26; padding: 25px; border-radius: 8px; border: 1px solid #1e2d3d; }
    
    .stButton>button { background-color: transparent; border: 1px solid #ef4444; color: #ef4444; border-radius: 6px; font-weight: bold; width: 100%; transition: all 0.3s ease; }
    .stButton>button:hover { background-color: rgba(239, 68, 68, 0.1); }
    
    .alert-high { background-color: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.3); padding: 15px; border-radius: 6px; color: #ef4444; font-weight: 600; margin-bottom: 15px; }
    .alert-low { background-color: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.3); padding: 15px; border-radius: 6px; color: #10b981; font-weight: 600; margin-bottom: 15px; }
    
    .stProgress > div > div > div > div { background-color: #00ffd0; }
    
    .metric-card { background-color: #101c26; padding: 20px; border-radius: 8px; border: 1px solid #1e2d3d; height: 100%; }
    .metric-title { color: #829ab1; font-size: 14px; margin-bottom: 10px; }
    .metric-value { color: #ffffff; font-size: 32px; font-weight: bold; margin: 0; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="header-container">
            <span class="header-icon">🛡️</span>
            <h1 class="header-text">FRAUD SHIELD ENTERPRISE</h1>
        </div>
        <p class="sub-text">Dynamic Real-Time Risk Intelligence Engine</p>
    """, unsafe_allow_html=True)
    
    model = get_model()
    
    tab1, tab2 = st.tabs(["Manual Scan", "Live Network"])
    
    with tab1:
        c1, c2 = st.columns([1, 1.8], gap="large")
        
        with c1:
            st.markdown('<div class="input-card">', unsafe_allow_html=True)
            with st.form("manual_scan_form"):
                amt = st.number_input("Amount", value=10000000000.00, step=1000.0, format="%.2f")
                age = st.slider("Age", 18, 90, 56)
                loc = st.selectbox("Location", ["California", "New York", "London", "Online", "Tokyo", "Singapore"])
                cat = st.selectbox("Category", ["Crypto", "Retail", "Electronics", "Entertainment", "Travel"])
                st.markdown("<br>", unsafe_allow_html=True)
                submit = st.form_submit_button("Analyze")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with c2:
            if submit:
                df_input = pd.DataFrame([[amt, int(time.time()%86400), age, loc, cat]], columns=["Amount", "Time", "CardHolderAge", "Location", "MerchantCategory"])
                risk = model.predict_proba(df_input)[0][1]
                
                st.session_state.manual_history.append(risk)
                if len(st.session_state.manual_history) > 25:
                    st.session_state.manual_history.pop(0)
                
                st.progress(risk)
                
                if risk >= 0.5:
                    st.markdown(f'<div class="alert-high">High Risk {risk*100:.2f}%</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="alert-low">Low Risk {risk*100:.2f}%</div>', unsafe_allow_html=True)
                    
            if len(st.session_state.manual_history) > 0:
                st.line_chart(st.session_state.manual_history, height=300)
            else:
                st.line_chart([0, 0], height=300)

    with tab2:
        is_live = st.toggle("Activate Live Monitoring")
        
        chart_placeholder = st.empty()
        metrics_placeholder = st.empty()
        
        if is_live:
            while is_live:
                btc_price = get_live_price()
                if not btc_price:
                    btc_price = 69113.20 + random.uniform(-50, 50)
                
                trade_val = btc_price * random.uniform(0.001, 0.01)
                
                df_live = pd.DataFrame([[trade_val, int(time.time()%86400), random.randint(20,60), "Online", "Crypto"]], columns=["Amount", "Time", "CardHolderAge", "Location", "MerchantCategory"])
                live_risk = model.predict_proba(df_live)[0][1]
                
                st.session_state.live_history.append(live_risk)
                if len(st.session_state.live_history) > 35:
                    st.session_state.live_history.pop(0)
                    
                with chart_placeholder.container():
                    st.line_chart(st.session_state.live_history, height=350)
                    
                with metrics_placeholder.container():
                    m1, m2, m3 = st.columns(3)
                    m1.markdown(f'<div class="metric-card"><div class="metric-title">USD Value</div><p class="metric-value">${trade_val:,.2f}</p></div>', unsafe_allow_html=True)
                    m2.markdown(f'<div class="metric-card"><div class="metric-title">BTC Price</div><p class="metric-value">${btc_price:,.2f}</p></div>', unsafe_allow_html=True)
                    m3.markdown(f'<div class="metric-card"><div class="metric-title">Risk</div><p class="metric-value">{live_risk*100:.2f}%</p></div>', unsafe_allow_html=True)
                
                time.sleep(1.5)
        else:
            with chart_placeholder.container():
                st.line_chart(st.session_state.live_history if len(st.session_state.live_history) > 0 else [0, 0], height=350)
            with metrics_placeholder.container():
                m1, m2, m3 = st.columns(3)
                m1.markdown('<div class="metric-card"><div class="metric-title">USD Value</div><p class="metric-value">$0.00</p></div>', unsafe_allow_html=True)
                m2.markdown('<div class="metric-card"><div class="metric-title">BTC Price</div><p class="metric-value">$0.00</p></div>', unsafe_allow_html=True)
                m3.markdown('<div class="metric-card"><div class="metric-title">Risk</div><p class="metric-value">0.00%</p></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
