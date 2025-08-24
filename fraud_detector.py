import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

try:
    df = pd.read_csv("credit_card_fraud_dataset_modified - credit_card_fraud_dataset_modified.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: The CSV file was not found. Please ensure it's in the correct directory.")
    exit()

print("\n--- Data Preprocessing ---")

df = df.drop(columns=['TransactionID'])

categorical_features = ['Location', 'MerchantCategory']
numerical_features = ['Amount', 'Time', 'CardHolderAge']

for col in numerical_features:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)

for col in categorical_features:
    if df[col].isnull().any():
        df[col].fillna(df[col].mode()[0], inplace=True)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

X = df.drop('IsFraud', axis=1)
y = df['IsFraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\n--- Model Development ---")

lr_pipeline = ImbPipeline(steps=[('preprocessor', preprocessor),
                                 ('smote', SMOTE(random_state=42)),
                                 ('classifier', LogisticRegression(random_state=42, solver='liblinear'))])

rf_pipeline = ImbPipeline(steps=[('preprocessor', preprocessor),
                                 ('smote', SMOTE(random_state=42)),
                                 ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))])

print("\nTraining Logistic Regression model...")
lr_pipeline.fit(X_train, y_train)
print("Logistic Regression training complete.")

print("\nTraining Random Forest model...")
rf_pipeline.fit(X_train, y_train)
print("Random Forest training complete.")

print("\n--- Performance Evaluation ---")

y_pred_lr = lr_pipeline.predict(X_test)
y_prob_lr = lr_pipeline.predict_proba(X_test)[:, 1]

y_pred_rf = rf_pipeline.predict(X_test)
y_prob_rf = rf_pipeline.predict_proba(X_test)[:, 1]

print("\n--------------------------------------")
print("Logistic Regression (Baseline) Report:")
print("--------------------------------------")
print(classification_report(y_test, y_pred_lr))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob_lr):.4f}")
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues')
plt.title('Logistic Regression Confusion Matrix')
plt.show()

print("\n--------------------------------------")
print("Random Forest (Chosen Model) Report:")
print("--------------------------------------")
print(classification_report(y_test, y_pred_rf))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob_rf):.4f}")
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens')
plt.title('Random Forest Confusion Matrix')
plt.show()

print("\n--- Comparison & Conclusion ---")
print(f"Logistic Regression ROC-AUC: {roc_auc_score(y_test, y_prob_lr):.4f}")
print(f"Random Forest ROC-AUC: {roc_auc_score(y_test, y_prob_rf):.4f}")
