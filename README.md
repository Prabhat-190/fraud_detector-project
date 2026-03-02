# 🏦 FRAUD_SHIELD / AUDIT_PRO  
AI-Based Transaction Risk Detection System  

🌐 **Live Application:**  
👉 https://frauddetectorpy-hx8n54x28xuyzes4jrei9b.streamlit.app/

---

## 📌 About This Project

FRAUD_SHIELD is a machine learning based fraud detection system that analyzes transaction details and predicts whether a transaction is likely to be fraudulent or legitimate.

The objective of this project was to simulate a real-world fintech risk monitoring dashboard using a structured ML pipeline and a professional user interface.

It combines:

- Backend ML engineering  
- Data preprocessing & imbalance handling  
- Interactive Streamlit dashboard  

---

## 🧠 How It Works

The system follows a complete ML workflow:

- Data cleaning and preprocessing  
- Handling missing values  
- Encoding categorical features  
- Scaling numerical features  
- Handling class imbalance using **SMOTE**  
- Training a **Random Forest classifier**  
- Probability-based fraud scoring  

The entire preprocessing and modeling logic is bundled inside a single pipeline using `ImbPipeline`, ensuring consistent transformations during prediction.

---

## ⚙️ Model Details

- **Algorithm:** Random Forest Classifier  
- **Imbalance Handling:** SMOTE (Synthetic Minority Oversampling)  
- **Train/Test Split:** Stratified sampling  
- **Output:** Fraud probability score  
- **Decision Logic:** Threshold-based classification  

The system outputs a fraud probability which is visualized as a risk index inside the dashboard.

---

## 🎨 User Interface

The UI is designed with a fintech-inspired dark theme.

### Key Features:

- Clean dashboard layout  
- Interactive form-based input  
- Real-time fraud probability display  
- Risk progress bar visualization  
- Clear alert status (Secure / High Risk)  

The goal was to create a system that feels closer to an enterprise monitoring tool rather than a basic ML demo.

---

## 🛠 Tech Stack

- Python  
- Streamlit  
- Scikit-learn  
- Imbalanced-learn (SMOTE)  
- Pandas  
- NumPy  

---

## 📂 Project Structure

```
FraudShield-AI
│
├── fraud_detector.py
├── credit_card_fraud_dataset_modified - credit_card_fraud_dataset_modified.csv
├── requirements.txt
└── README.md
```

---

## ▶ How to Run Locally

1. Clone the repository:
```
git clone https://github.com/Prabhat-190/FraudSheild-AI.git
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run the application:
```
streamlit run fraud_detector.py
```

---

## 📊 Sample Output

The model returns:

- Fraud Probability (%)
- Risk Status (Secure / High Risk)
- Visual risk progress indicator

---

## 🚀 Future Improvements

- Threshold optimization using Precision-Recall tradeoff  
- ROC-AUC performance visualization  
- Real-world dataset integration  
- API-based deployment architecture  

---

If you found this project interesting, feel free to ⭐ the repository.
