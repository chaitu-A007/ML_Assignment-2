#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

# Page Config
st.set_page_config(page_title="Online Shopper Intention Predictor", layout="wide")

st.title("🛍️ Online Shopper Intention Classification")
st.write("This app demonstrates 6 different ML models to predict customer purchase intent.")

# --- STEP 1: DATASET UPLOAD ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your test CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded successfully!")
    
    # Preprocessing (Standard for all models)
    le = LabelEncoder()
    for col in ['Month', 'VisitorType', 'Weekend', 'Revenue']:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
    
    X = df.drop('Revenue', axis=1) if 'Revenue' in df.columns else df
    y = df['Revenue'] if 'Revenue' in df.columns else None

    # --- STEP 2: MODEL SELECTION--
    st.sidebar.header("2. Model Settings")
    model_option = st.sidebar.selectbox(
        "Select Model",
        ("Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest", "XGBoost")
    )

    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Scaling for distance-based models
        if model_option in ["Logistic Regression", "kNN"]:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Initialize Models
        if model_option == "Logistic Regression":
            model = LogisticRegression()
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

        # Train and Predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1]

        # --- STEP 3: DISPLAY METRICS  ---
        st.subheader(f"📊 {model_option} Performance Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
        col2.metric("AUC Score", f"{roc_auc_score(y_test, y_probs):.4f}")
        col3.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
        
        col4, col5, col6 = st.columns(3)
        col4.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
        col5.metric("F1 Score", f"{f1_score(y_test, y_pred):.4f}")
        col6.metric("MCC Score", f"{matthews_corrcoef(y_test, y_pred):.4f}")

        # --- STEP 4: VISUALIZATION ---
        st.subheader("📋 Analysis Results")
        tab1, tab2 = st.tabs(["Confusion Matrix", "Classification Report"])
        
        with tab1:
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot(fig)
            
        with tab2:
            st.text("Detailed Classification Report:")
            st.text(classification_report(y_test, y_pred))

else:
    st.info("Please upload a CSV file from the sidebar to begin.")
    st.image("https://raw.githubusercontent.com/sharmaroshan/Online-Shoppers-Purchasing-Intention/master/online_shoppers_intention.csv", caption="Standard Dataset Example")


# In[ ]:




