import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef, 
                             confusion_matrix, classification_report)

# Page Configuration
st.set_page_config(page_title="Shopper Intent Classifier", layout="wide")

st.title("🛍️ Online Shopper Intention Classification")
st.write("Upload your dataset to evaluate 6 different ML models or download the training data below.")

# --- STEP 1: DATASET UPLOAD & DOWNLOAD ---
st.sidebar.header("1. Data Operations")

# Download Section: This allows the evaluator to download your full CSV file
st.sidebar.subheader("Download Dataset")
try:
    # This looks for the file in your GitHub repository folder
    with open("online_shoppers_intention.csv", "rb") as file:
        st.sidebar.download_button(
            label="📥 Download Full CSV",
            data=file,
            file_name='online_shoppers_intention.csv',
            mime='text/csv',
            help="Download the full dataset with 12,330 records to test the app."
        )
except FileNotFoundError:
    st.sidebar.warning("Note: Place 'online_shoppers_intention.csv' in your GitHub root to enable local download button.")
    st.sidebar.markdown("[🔗 Download from Source](https://raw.githubusercontent.com/sharmaroshan/Online-Shoppers-Purchasing-Intention/master/online_shoppers_intention.csv)")

st.sidebar.divider()

# Upload Section
uploaded_file = st.sidebar.file_uploader("Upload Test CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File Uploaded Successfully!")
    
    # Simple Preprocessing
    le = LabelEncoder()
    categorical_cols = ['Month', 'VisitorType', 'Weekend', 'Revenue']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
    
    if 'Revenue' in df.columns:
        X = df.drop('Revenue', axis=1)
        y = df['Revenue']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # --- STEP 2: MODEL SELECTION ---
        st.sidebar.header("2. Model Selection")
        model_option = st.sidebar.selectbox(
            "Choose a Model",
            ("Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest", "XGBoost")
        )

        # Model Initialization
        if model_option == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_option == "Decision Tree":
            model = DecisionTreeClassifier(max_depth=5)
        elif model_option == "kNN":
            model = KNeighborsClassifier()
        elif model_option == "Naive Bayes":
            model = GaussianNB()
        elif model_option == "Random Forest":
            model = RandomForestClassifier(n_estimators=100)
        elif model_option == "XGBoost":
            model = XGBClassifier(n_estimators=50, max_depth=3,eval_metric='logloss')

        # Scaling for distance-based models
        if model_option in ["Logistic Regression", "kNN"]:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Train and Predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1]

        # --- STEP 3: DISPLAY METRICS ---
        st.subheader(f"📊 {model_option} Evaluation Metrics")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
        m2.metric("AUC", f"{roc_auc_score(y_test, y_probs):.3f}")
        m3.metric("Precision", f"{precision_score(y_test, y_pred):.3f}")
        m4.metric("Recall", f"{recall_score(y_test, y_pred):.3f}")
        m5.metric("F1", f"{f1_score(y_test, y_pred):.3f}")
        m6.metric("MCC", f"{matthews_corrcoef(y_test, y_pred):.3f}")

        # --- STEP 4: VISUALIZATION ---
        st.divider()
        col_left, col_right = st.columns([1, 1.5])

        with col_left:
            st.write("### Confusion Matrix")
            # Optimized size for professional UI
            fig, ax = plt.subplots(figsize=(4, 3)) 
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            st.pyplot(fig, use_container_width=False) 

        with col_right:
            st.write("### Classification Report")
            st.text(classification_report(y_test, y_pred))

    else:
        st.error("The CSV must contain a 'Revenue' target column for evaluation.")
else:
    st.info("Waiting for CSV upload... Please upload the dataset in the sidebar to view results.")