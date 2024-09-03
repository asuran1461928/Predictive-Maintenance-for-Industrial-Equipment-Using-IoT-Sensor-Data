import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Load the AI4I 2020 dataset
df = pd.read_csv('sensor_data.csv')

# Set Streamlit page configuration
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

# Sidebar - User input features
st.sidebar.header('Input Sensor Data')
air_temp = st.sidebar.number_input('Air Temperature [K]', min_value=290.0, max_value=320.0, value=300.0)
process_temp = st.sidebar.number_input('Process Temperature [K]', min_value=305.0, max_value=320.0, value=310.0)
rotational_speed = st.sidebar.number_input('Rotational Speed [rpm]', min_value=1000, max_value=3000, value=1500)
torque = st.sidebar.number_input('Torque [Nm]', min_value=0.0, max_value=100.0, value=50.0)
tool_wear = st.sidebar.number_input('Tool Wear [min]', min_value=0, max_value=300, value=150)
machine_type = st.sidebar.selectbox('Machine Type', ['L', 'M', 'H'])

# Encode 'Machine Type' input
type_encoded = {'L': 0, 'M': 1, 'H': 2}[machine_type]

# Create DataFrame for input
input_data = pd.DataFrame([[type_encoded, air_temp, process_temp, rotational_speed, torque, tool_wear]], 
                          columns=['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'])

# Add missing columns
missing_columns = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
for col in missing_columns:
    input_data[col] = 0

# Data Overview
st.title("Predictive Maintenance Dashboard")
st.subheader("Dataset Overview")
st.write("Preview of the dataset:")
st.dataframe(df.head())
st.write("Basic statistics of the dataset:")
st.write(df.describe())

# Data Preprocessing
df.drop(['UDI', 'Product ID'], axis=1, inplace=True)
df['Type'] = df['Type'].astype('category').cat.codes
X = df.drop('Machine failure', axis=1)
y = df['Machine failure']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training and Evaluation
st.subheader("Model Training and Evaluation")
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred = rf_model.predict(X_test_scaled)
y_prob = rf_model.predict_proba(X_test_scaled)[:, 1]

st.write("Classification Report:")
st.text(classification_report(y_test, y_pred))
st.write(f"AUC-ROC Score: {roc_auc_score(y_test, y_prob):.2f}")

cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
st.write("Confusion Matrix:")
st.pyplot(fig_cm)

fpr, tpr, _ = roc_curve(y_test, y_prob)
fig_roc, ax_roc = plt.subplots()
ax_roc.plot(fpr, tpr, marker='.', label='Random Forest')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.legend()
st.write("ROC Curve:")
st.pyplot(fig_roc)

# Real-time Prediction
st.sidebar.subheader('Real-time Prediction')
if st.sidebar.button('Predict Machine Failure'):
    input_data_scaled = scaler.transform(input_data)
    failure_prob = rf_model.predict_proba(input_data_scaled)[:, 1]
    st.sidebar.write(f"Predicted Probability of Machine Failure: {failure_prob[0]:.2f}")

# Data Visualization
st.subheader("Data Visualization")
st.write("Distribution of Features")
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))
sns.histplot(df['Air temperature [K]'], kde=True, ax=axes[0, 0])
sns.histplot(df['Process temperature [K]'], kde=True, ax=axes[0, 1])
sns.histplot(df['Rotational speed [rpm]'], kde=True, ax=axes[1, 0])
sns.histplot(df['Torque [Nm]'], kde=True, ax=axes[1, 1])
sns.histplot(df['Tool wear [min]'], kde=True, ax=axes[2, 0])
sns.countplot(df['Type'], ax=axes[2, 1])
st.pyplot(fig)

# Correlation Heatmap
st.write("Correlation Heatmap")
fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
st.pyplot(fig_corr)
