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

# --- NEW: Helper function to convert DF to CSV for download ---
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

st.title("🛍️ Online Shopper Intention Classification")
st.write("Upload your dataset to evaluate 6 different ML models.")

# --- STEP 1: DATASET UPLOAD & DOWNLOAD ---
st.sidebar.header("1. Data Operations")

# --- NEW: Download Sample Section ---
st.sidebar.subheader("Need a sample file?")
# Creating a dummy sample based on your dataset structure
sample_data = pd.DataFrame({
    'Administrative': [0, 1], 'Administrative_Duration': [0.0, 10.5],
    'Informational': [0, 2], 'Informational_Duration': [0.0, 15.0],
    'ProductRelated': [1, 20], 'ProductRelated_Duration': [0.0, 500.0],
    'BounceRates': [0.2, 0.01], 'ExitRates': [0.2, 0.02],
    'PageValues': [0.0, 15.0], 'SpecialDay': [0.0, 0.0],
    'Month': ['Feb', 'Nov'], 'OperatingSystems': [1, 2],
    'Browser': [1, 2], 'Region': [1, 3], 'TrafficType': [1, 2],
    'VisitorType': ['Returning_Visitor', 'New_Visitor'],
    'Weekend': [False, True], 'Revenue': [False, True]
})
csv_sample = convert_df(sample_data)

st.sidebar.download_button(
    label="📥 Download Sample CSV",
    data=csv_sample,
    file_name='online_shoppers_sample.csv',
    mime='text/csv',
)
st.sidebar.divider()

# Original File Uploader
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

        # Model Initialization (Remaining code stays the same...)
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
            model = XGBClassifier(eval_metric='logloss')

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
    st.info("Waiting for CSV upload... Please upload the test dataset in the sidebar.")