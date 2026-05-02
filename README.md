# 🧬 Cancer Detection using Machine Learning (Binary Classification)

This project applies machine learning techniques to classify whether a patient has cancer or not using a dataset of patient attributes. It compares two models — **Perceptron** and **Logistic Regression** — and evaluates their performance using standard metrics and confusion matrix visualization.

---

## 📌 Features

- Data preprocessing and cleaning
- Conversion of multi-level target into binary classification
- Label encoding for categorical features
- Train-test split for evaluation
- Two ML models:
  - Perceptron
  - Logistic Regression
- Accuracy, confusion matrix, and classification report
- Heatmap visualization using Seaborn
- Custom prediction on sample data

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- OpenPyXL (for Excel file handling)

---

## 📂 Dataset

- Input file: `cancer patient data sets.xlsx`
- Contains patient-related features and cancer level

### 🎯 Target Variable

Original `Level` column is converted to binary:

- `0` → No Cancer  
- `1` → Cancer  

---

## ⚙️ Data Preprocessing

- Removed unnecessary columns (e.g., Patient ID)
- Converted categorical data using Label Encoding
- Transformed target into binary classification
- Checked for missing values

---

## 🧠 Models Used

### 1️⃣ Perceptron
- Linear classifier
- Fast and simple baseline model

### 2️⃣ Logistic Regression
- Probabilistic linear model
- More stable and commonly used for classification

---

## 🚀 How to Run

### ▶️ Step 1: Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
