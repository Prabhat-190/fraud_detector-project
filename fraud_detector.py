import os
import time
import random
import math
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
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

MODEL_FILE = "fraud_model_final_v14.pkl"

def generate_dynamic_dataset():
    n = 20000
    amounts = np.random.uniform(5.0, 80000.0, size=n)
    times = np.random.randint(0, 86400, size=n)
    ages = np.random.randint(18, 85, size=n)
    locations = np.random.choice(["California", "New York", "London", "Online", "Tokyo"], size=n)
    cats = np.random.choice(["Retail", "Electronics", "Crypto", "Entertainment"], size=n)

    z = (
        -2.5
        + 0.00005 * amounts
        + 0.00000001 * (amounts ** 2)
        + 0.00002 * times
        - 0.015 * ages
    )

    z += np.where(cats == "Crypto", 1.8, 0)
    z += np.where(cats == "Electronics", 0.7, 0)
    z -= np.where(cats == "Retail", 1.2, 0)
    z += np.where(locations == "Online", 1.5, 0)

    noise = np.random.normal(0, 1.2, size=n)
    z += noise

    probs = 1 / (1 + np.exp(-z))
    is_fraud = np.random.binomial(1, probs)

    return pd.DataFrame({
        "Amount": amounts,
        "Time": times,
        "CardHolderAge": ages,
        "Location": locations,
        "MerchantCategory": cats,
        "IsFraud": is_fraud
    })

def build_model():
    df = generate_dynamic_dataset()
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
        ("classifier", RandomForestClassifier(n_estimators=400, min_samples_split=5, random_state=42))
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
    return build_model()

def calculate_realtime_risk(input_df, model):
    base_prob = model.predict_proba(input_df)[0][1]
    amount = float(input_df["Amount"].values[0])
    age = int(input_df["CardHolderAge"].values[0])
    time_val = int(input_df["Time"].values[0])
    location = str(input_df["Location"].values[0])
    category = str(input_df["MerchantCategory"].values[0])

    amount_score = min(1.0, math.log1p(amount) / 12.0)
    hour = time_val // 3600
    time_score = 0.7 if (hour < 5 or hour > 23) else 0.2
    age_score = max(0.0, (30 - age) / 30.0)
    cat_score = 0.8 if category == "Crypto" else 0.4 if category == "Electronics" else 0.1
    loc_score = 0.6 if location == "Online" else 0.2
    noise = random.uniform(-0.04, 0.04)

    final = (
        0.45 * base_prob +
        0.18 * amount_score +
        0.10 * time_score +
        0.10 * age_score +
        0.07 * cat_score +
        0.05 * loc_score
    ) + noise

    return float(np.clip(final, 0.0, 1.0))

def fetch_price_ccxt(symbol="BTC/USD"):
    if not CCXT_AVAILABLE:
        return None
    try:
        exchange = ccxt.kraken()
        ticker = exchange.fetch_ticker(symbol)
        price = ticker.get("last") or ticker.get("close") or None
        if price is None:
            return None
        return float(price)
    except:
        return None

def simulate_price(prev_price):
    drift = 0
    vol = max(1.0, prev_price * 0.002)
    change = random.gauss(drift, vol * 0.001)
    new_price = max(10.0, prev_price + change)
    return float(new_price)

def main():
    st.set_page_config(page_title="Fraud Shield Enterprise", page_icon="🛡️", layout="wide")
    if "risk_history" not in st.session_state:
        st.session_state.risk_history = []
    if "last_price" not in st.session_state:
        st.session_state.last_price = 60000.0
    if "fetch_failures" not in st.session_state:
        st.session_state.fetch_failures = 0
    st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg,#061024,#0f2633); color: #e6f7f2; }
    .big-title { font-size: 44px; font-weight: 800; margin-bottom:6px;
                 background: linear-gradient(90deg,#00ffd0,#00b3ff);
                 -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
    .sub { color: #bfeee6; margin-top: -6px; margin-bottom: 10px; }
    .card { background: rgba(255,255,255,0.03); padding: 14px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.04); }
    .metric { font-size: 18px; color: #00ffd0; }
    .large-metric { font-size: 30px; font-weight:700; color: #00ffd0; }
    .status-green { color: #00ff99; font-weight:700; }
    .status-red { color: #ff6b6b; font-weight:700; }
    .footer { color: #9fded1; font-size: 12px; margin-top:8px; }
    .stButton>button { border-radius: 10px; height:42px; background: linear-gradient(90deg,#00ffd0,#00b3ff); color: black; font-weight:700; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="big-title">🛡️ Fraud Shield Enterprise</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">Real-time fraud risk monitoring · live feed · adaptive scoring</div>', unsafe_allow_html=True)
    model = get_model()
    left, right = st.columns([1.2, 2.2])
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Manual Transaction")
        with st.form("manual_form"):
            amt = st.number_input("Amount (USD)", min_value=0.0, value=250.0, step=1.0)
            age = st.slider("Cardholder Age", 18, 90, 30)
            loc = st.selectbox("Location", ["California", "New York", "London", "Online", "Tokyo"])
            cat = st.selectbox("Merchant Category", ["Retail", "Electronics", "Crypto", "Entertainment"])
            submit = st.form_submit_button("Analyze")
        if submit:
            df_input = pd.DataFrame([[amt, int(time.time() % 86400), age, loc, cat]],
                                    columns=["Amount","Time","CardHolderAge","Location","MerchantCategory"])
            risk = calculate_realtime_risk(df_input, model)
            st.session_state.risk_history.append(risk)
            if len(st.session_state.risk_history) > 120:
                st.session_state.risk_history.pop(0)
            st.markdown(f"<div style='margin-top:8px'><span class='large-metric'>{risk*100:.2f}%</span></div>", unsafe_allow_html=True)
            if risk > 0.6:
                st.error("Action: Recommend HOLD / Block")
            else:
                st.success("Action: Proceed / Approve")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div style="height:12px"></div>')
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Controls & Status")
        live_toggle = st.toggle("Live Network Monitoring", key="live_toggle")
        st.markdown(f"<div style='margin-top:8px'><span class='metric'>Model: </span><span style='color:#dfffe7'>Random Forest (cached)</span></div>", unsafe_allow_html=True)
        last_sync = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        st.markdown(f"<div style='margin-top:6px'><span class='metric'>Last Sync:</span> <span style='color:#bfeee6'>{last_sync}</span></div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Live Network · BTC/USD")
        if live_toggle:
            st_autorefresh(interval=1000, key="live_refresh")
            price = fetch_price_ccxt("BTC/USD")
            if price is None:
                st.session_state.fetch_failures += 1
                if st.session_state.fetch_failures >= 3:
                    price = simulate_price(st.session_state.last_price)
                else:
                    price = simulate_price(st.session_state.last_price * (1 + random.uniform(-0.0005,0.0005)))
            else:
                st.session_state.fetch_failures = 0
            st.session_state.last_price = price
            df_live = pd.DataFrame([[price, int(time.time() % 86400), random.randint(20,65), "Online", "Crypto"]],
                                   columns=["Amount","Time","CardHolderAge","Location","MerchantCategory"])
            risk_live = calculate_realtime_risk(df_live, model)
            st.session_state.risk_history.append(risk_live)
            if len(st.session_state.risk_history) > 120:
                st.session_state.risk_history.pop(0)
            c1, c2, c3 = st.columns([1.2,1.2,1])
            c1.metric("BTC Price (USD)", f"${price:,.2f}")
            c2.metric("Risk", f"{risk_live*100:.2f}%")
            status = "<span class='status-green'>ACTIVE</span>" if st.session_state.fetch_failures < 4 else "<span class='status-red'>FALLBACK</span>"
            c3.markdown(status, unsafe_allow_html=True)
            st.progress(risk_live)
            chart_df = pd.DataFrame({"Risk": st.session_state.risk_history})
            st.line_chart(chart_df, height=320)
            st.markdown('<div class="footer">Live feed uses Kraken when available; otherwise simulated fallback keeps the stream healthy.</div>', unsafe_allow_html=True)
        else:
            st.info("Live monitoring is off. Toggle to start live checks.")
            st.markdown('<div style="height:320px"></div>')
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<div style='margin-top:12px; color:#bfeee6'>Tip: If live pricing is unreliable due to exchange rate limits, enable fallback in settings (currently automatic after 3 failures).</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
