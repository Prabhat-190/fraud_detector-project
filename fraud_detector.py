import os
import time
import datetime
import random
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
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

MODEL_FILE = "fraud_model_v22.pkl"

def generate_fluid_data():
    np.random.seed(42)
    n = 15000
    amounts = np.random.uniform(10.0, 100000.0, size=n)
    times = np.random.randint(0, 86400, size=n)
    ages = np.random.randint(18, 80, size=n)
    locations = np.random.choice(["California", "New York", "London", "Online", "Tokyo"], size=n)
    cats = np.random.choice(["Retail", "Electronics", "Crypto", "Entertainment"], size=n)
    
    z = (amounts / 50000.0)
    z += np.where((times < 18000) | (times > 79200), 0.15, 0.0)
    z += np.where(ages < 30, 0.1, 0.0)
    z += np.where((cats == "Crypto") & (locations == "Online"), 0.25, 0.0)
    
    probs = 1 / (1 + np.exp(-(z - 0.5) * 5))
    is_fraud = np.random.binomial(1, np.clip(probs, 0.01, 0.99))
    
    return pd.DataFrame({"Amount": amounts, "Time": times, "CardHolderAge": ages, "Location": locations, "MerchantCategory": cats, "IsFraud": is_fraud})

def build_model():
    df = generate_fluid_data()
    cat_cols = ["Location", "MerchantCategory"]
    num_cols = ["Amount", "Time", "CardHolderAge"]
    preprocessor = ColumnTransformer([("num", StandardScaler(), num_cols), ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])
    X = df.drop("IsFraud", axis=1)
    y = df["IsFraud"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline = Pipeline([
        ("preprocessor", preprocessor), 
        ("classifier", RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=2, random_state=42))
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
    return build_model()

def get_live_price():
    if not CCXT_AVAILABLE:
        return None
    try:
        exchange = ccxt.kraken()
        ticker = exchange.fetch_ticker("BTC/USD")
        return float(ticker.get("last")) if ticker.get("last") else None
    except Exception:
        return None

def create_gauge(risk_value):
    color = "#ef4444" if risk_value >= 0.5 else "#00ffd0"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_value * 100,
        number={'suffix': "%", 'font': {'color': "#ffffff", 'size': 36}},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "RISK INDEX", 'font': {'color': "#8a9ba8", 'size': 14}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#1e2d3d"},
            'bar': {'color': color},
            'bgcolor': "#121d28",
            'borderwidth': 2,
            'bordercolor': "#1c2b3a",
            'steps': [
                {'range': [0, 50], 'color': "rgba(0, 255, 208, 0.05)"},
                {'range': [50, 100], 'color': "rgba(239, 68, 68, 0.05)"}
            ],
            'threshold': {
                'line': {'color': "#ef4444", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#8a9ba8", 'family': "sans-serif"},
        height=250,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    return fig

def main():
    st.set_page_config(page_title="Fraud Shield", layout="wide")
    
    if "manual_history" not in st.session_state:
        st.session_state.manual_history = pd.DataFrame(columns=["Risk"])
    if "live_history" not in st.session_state:
        st.session_state.live_history = pd.DataFrame(columns=["Risk"])
    if "alert_ledger" not in st.session_state:
        st.session_state.alert_ledger = pd.DataFrame(columns=["Time", "Amount", "Location", "Category", "Risk", "Status"])
    if "last_manual_risk" not in st.session_state:
        st.session_state.last_manual_risk = None
        
    st.markdown("""
    <style>
    .stApp { background: linear-gradient(180deg, #07131b 0%, #0d1e2a 100%); color: #ffffff; }
    
    .header-box { display: flex; align-items: center; margin-bottom: 0; padding-bottom: 0; }
    .header-icon { font-size: 40px; color: #00ffd0; margin-right: 15px; }
    .header-text { font-size: 42px; font-weight: 900; color: #00ffd0; margin: 0; letter-spacing: 0.5px; }
    .sub-text { color: #8a9ba8; font-size: 13px; margin-top: 5px; margin-bottom: 30px; }
    
    div[data-testid="stTabs"] button { color: #8a9ba8 !important; font-size: 16px !important; font-weight: 600 !important; background-color: transparent !important; }
    div[data-testid="stTabs"] button[aria-selected="true"] { color: #ef4444 !important; border-bottom: 2px solid #ef4444 !important; }
    
    .input-card { background-color: #121d28; padding: 25px; border-radius: 6px; border: 1px solid #1c2b3a; margin-bottom: 20px;}
    .metric-card { background-color: #121d28; padding: 20px; border-radius: 6px; border: 1px solid #1c2b3a; height: 100%; box-shadow: 0 4px 6px rgba(0,0,0,0.2); }
    .metric-label { color: #8a9ba8; font-size: 13px; font-weight: 500; margin-bottom: 8px; }
    .metric-val { color: #ffffff; font-size: 32px; font-weight: 700; margin: 0; }
    
    .alert-high { background-color: #2b171c; border: 1px solid rgba(239,68,68,0.3); padding: 15px; border-radius: 4px; color: #ef4444; font-size: 15px; font-weight: 600; margin-bottom: 15px; }
    .alert-low { background-color: #122622; border: 1px solid rgba(0,255,208,0.3); padding: 15px; border-radius: 4px; color: #00ffd0; font-size: 15px; font-weight: 600; margin-bottom: 15px; }
    
    .stProgress > div > div > div > div { background-color: #00ffd0; }
    
    div[data-testid="stFormSubmitButton"] > button, div[data-testid="stButton"] > button { 
        background: linear-gradient(180deg, #ff4b4b 0%, #b91c1c 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        width: 100% !important;
        padding: 14px 0 !important;
        transition: all 0.1s ease-in-out !important;
        box-shadow: 0 6px 0 #7f1d1d, 0 8px 15px rgba(0,0,0,0.4) !important;
    }
    div[data-testid="stFormSubmitButton"] > button:hover, div[data-testid="stButton"] > button:hover { 
        transform: translateY(2px) !important;
        box-shadow: 0 4px 0 #7f1d1d, 0 5px 10px rgba(0,0,0,0.4) !important;
        background: linear-gradient(180deg, #ff6b6b 0%, #dc2626 100%) !important;
    }
    div[data-testid="stFormSubmitButton"] > button:active, div[data-testid="stButton"] > button:active { 
        transform: translateY(6px) !important;
        box-shadow: 0 0 0 #7f1d1d, 0 2px 5px rgba(0,0,0,0.4) !important;
        background: #b91c1c !important;
    }
    div[data-testid="stFormSubmitButton"] p, div[data-testid="stButton"] p {
        font-weight: 900 !important;
        font-size: 16px !important;
        color: white !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        margin: 0 !important;
    }
    
    div[data-testid="stSlider"] > div > div > div > div { background-color: #ef4444 !important; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="header-box">
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
            with st.form("manual"):
                amt = st.number_input("Amount", value=10000.00, format="%.2f")
                age = st.slider("Age", 18, 90, 56)
                loc = st.selectbox("Location", ["California", "New York", "London", "Online", "Tokyo"])
                cat = st.selectbox("Category", ["Crypto", "Retail", "Electronics", "Entertainment"])
                st.markdown("<br>", unsafe_allow_html=True)
                sub = st.form_submit_button("ANALYZE RISK")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with c2:
            if sub:
                df_in = pd.DataFrame([[amt, int(time.time()%86400), age, loc, cat]], columns=["Amount", "Time", "CardHolderAge", "Location", "MerchantCategory"])
                risk = model.predict_proba(df_in)[0][1]
                st.session_state.last_manual_risk = risk
                new_row = pd.DataFrame({"Risk": [risk]})
                st.session_state.manual_history = pd.concat([st.session_state.manual_history, new_row], ignore_index=True).tail(30)
                
            risk_val = st.session_state.last_manual_risk
            
            if risk_val is not None:
                if risk_val >= 0.5:
                    st.markdown(f'<div class="alert-high">High Risk {risk_val*100:.2f}%</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="alert-low">Low Risk {risk_val*100:.2f}%</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-low" style="color:#8a9ba8; border-color:#1c2b3a; background:transparent;">Awaiting input parameters...</div>', unsafe_allow_html=True)

            g_col1, g_col2 = st.columns([1, 1.5])
            with g_col1:
                # ADDED UNIQUE KEY HERE
                st.plotly_chart(create_gauge(risk_val if risk_val else 0.0), use_container_width=True, config={'displayModeBar': False}, key="manual_gauge")
            with g_col2:
                if not st.session_state.manual_history.empty:
                    st.line_chart(st.session_state.manual_history, height=250)
                else:
                    st.line_chart([0], height=250)

    with tab2:
        is_live = st.toggle("Activate Live Monitoring")
        
        live_placeholder = st.empty()
        
        if is_live:
            while is_live:
                btc = get_live_price()
                if not btc:
                    btc = 69113.20 + random.uniform(-20, 20)
                
                val = btc * random.uniform(0.001, 0.005)
                loc_val = random.choice(["California", "New York", "London", "Online", "Tokyo"])
                cat_val = random.choice(["Crypto", "Retail", "Electronics"])
                
                df_live = pd.DataFrame([[val, int(time.time()%86400), random.randint(20,50), loc_val, cat_val]], columns=["Amount", "Time", "CardHolderAge", "Location", "MerchantCategory"])
                live_risk = model.predict_proba(df_live)[0][1]
                
                new_live_row = pd.DataFrame({"Risk": [live_risk]})
                st.session_state.live_history = pd.concat([st.session_state.live_history, new_live_row], ignore_index=True).tail(40)
                
                status_mark = "🚨 BLOCKED" if live_risk >= 0.5 else "✅ SECURE"
                new_ledger_row = pd.DataFrame([{
                    "Time": datetime.datetime.now().strftime("%H:%M:%S"),
                    "Amount": f"${val:,.2f}",
                    "Location": loc_val,
                    "Category": cat_val,
                    "Risk": f"{live_risk*100:.2f}%",
                    "Status": status_mark
                }])
                st.session_state.alert_ledger = pd.concat([new_ledger_row, st.session_state.alert_ledger], ignore_index=True).head(20)
                
                with live_placeholder.container():
                    m1, m2, m3 = st.columns(3)
                    m1.markdown(f'<div class="metric-card"><div class="metric-label">USD Value</div><p class="metric-val">${val:,.2f}</p></div>', unsafe_allow_html=True)
                    m2.markdown(f'<div class="metric-card"><div class="metric-label">BTC Price</div><p class="metric-val">${btc:,.2f}</p></div>', unsafe_allow_html=True)
                    m3.markdown(f'<div class="metric-card"><div class="metric-label">Risk</div><p class="metric-val">{live_risk*100:.2f}%</p></div>', unsafe_allow_html=True)
                    
                    v1, v2 = st.columns([1, 2])
                    with v1:
                        dynamic_key = f"gauge_{int(time.time() * 1000)}"
                        st.plotly_chart(create_gauge(live_risk), use_container_width=True, config={'displayModeBar': False}, key=dynamic_key)
                    with v2:
                        st.line_chart(st.session_state.live_history, height=250)
                        
                    st.markdown('<div class="input-card"><h3 style="color:#00ffd0; margin-top:0; font-size:18px;">Fraud Alert Dashboard (Last 20)</h3></div>', unsafe_allow_html=True)
                    
                    styled_ledger = st.session_state.alert_ledger.style.map(
                        lambda v: 'color: #ef4444; font-weight: bold;' if v == '🚨 BLOCKED' else ('color: #00ffd0; font-weight: bold;' if v == '✅ SECURE' else ''),
                        subset=['Status']
                    )
                    st.dataframe(styled_ledger, use_container_width=True, hide_index=True)
                
                time.sleep(1.5)
        else:
            with live_placeholder.container():
                m1, m2, m3 = st.columns(3)
                m1.markdown('<div class="metric-card"><div class="metric-label">USD Value</div><p class="metric-val">$0.00</p></div>', unsafe_allow_html=True)
                m2.markdown('<div class="metric-card"><div class="metric-label">BTC Price</div><p class="metric-val">$0.00</p></div>', unsafe_allow_html=True)
                m3.markdown('<div class="metric-card"><div class="metric-label">Risk</div><p class="metric-val">0.00%</p></div>', unsafe_allow_html=True)
                
                v1, v2 = st.columns([1, 2])
                with v1:
                    # ADDED UNIQUE KEY HERE
                    st.plotly_chart(create_gauge(0.0), use_container_width=True, config={'displayModeBar': False}, key="live_gauge_inactive")
                with v2:
                    st.line_chart(st.session_state.live_history if not st.session_state.live_history.empty else pd.DataFrame({"Risk": [0]}), height=250)
                    
                st.markdown('<div class="input-card"><h3 style="color:#00ffd0; margin-top:0; font-size:18px;">Fraud Alert Dashboard (Last 20)</h3></div>', unsafe_allow_html=True)
                st.dataframe(st.session_state.alert_ledger, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
