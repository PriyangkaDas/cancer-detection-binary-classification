# =========================================
# CANCER DATA CLASSIFICATION (BINARY - 2x2)
# =========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.preprocessing import LabelEncoder

# ==============================
# 1. LOAD DATA
# ==============PIP================
df = pd.read_excel(r"C:\Users\HP\Downloads\cancer patient data sets.xlsx")

# ==============================
# 2. DATA CLEANING
# ==============================
if "Patient Id" in df.columns:
    df = df.drop(columns=["Patient Id"])

# Convert Level to numeric first
df["Level"] = df["Level"].astype("category").cat.codes

# 🔥 CONVERT TO BINARY (IMPORTANT)
# 0 = No Cancer
# 1 = Cancer (Level 1 & 2)
df["Level"] = df["Level"].apply(lambda x: 0 if x == 0 else 1)

# Encode categorical columns
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

print("\nMissing values:\n", df.isnull().sum())

# ==============================
# 3. FEATURE SELECTION
# ==============================
X = df.drop(columns=["Level"])
y = df["Level"]

# ==============================
# 4. TRAIN-TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# ==============================
# 5. PERCEPTRON MODEL
# ==============================
perceptron = Perceptron(random_state=0)
perceptron.fit(X_train, y_train)

y_pred1 = perceptron.predict(X_test)
cm1 = confusion_matrix(y_test, y_pred1)

print("\n=== PERCEPTRON RESULTS ===")
print("Accuracy:", accuracy_score(y_test, y_pred1))
print("Confusion Matrix:\n", cm1)
print(classification_report(y_test, y_pred1))

# 🔥 HEATMAP (2x2)
plt.figure(figsize=(5,4))
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues',
            xticklabels=["No Cancer", "Cancer"],
            yticklabels=["No Cancer", "Cancer"])
plt.title("Perceptron Confusion Matrix (2x2)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==============================
# 6. LOGISTIC REGRESSION MODEL
# ==============================
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

y_pred2 = log_model.predict(X_test)
cm2 = confusion_matrix(y_test, y_pred2)

print("\n=== LOGISTIC REGRESSION RESULTS ===")
print("Accuracy:", accuracy_score(y_test, y_pred2))
print("Confusion Matrix:\n", cm2)
print(classification_report(y_test, y_pred2))

# 🔥 HEATMAP (2x2)
plt.figure(figsize=(5,4))
sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens',
            xticklabels=["No Cancer", "Cancer"],
            yticklabels=["No Cancer", "Cancer"])
plt.title("Logistic Regression Confusion Matrix (2x2)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==============================
# 7. CUSTOM PREDICTION
# ==============================
sample = X_test.iloc[0:1]
print("\nSample Prediction (0=No Cancer, 1=Cancer):", log_model.predict(sample))

