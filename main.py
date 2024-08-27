# Import necessary libraries
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

# Preview the data
st.write("Dataset Preview:")
st.write(df.head())

# Data Preprocessing
# Dropping 'UDI' and 'Product ID' columns as they are not useful for prediction
df.drop(['UDI', 'Product ID'], axis=1, inplace=True)

# Encoding 'Type' column (categorical) to numerical values
df['Type'] = df['Type'].astype('category').cat.codes

# Define features and target (assuming 'Machine failure' is the target)
X = df.drop('Machine failure', axis=1)
y = df['Machine failure']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build and train the model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Make predictions and evaluate the model
y_pred = rf_model.predict(X_test_scaled)
y_prob = rf_model.predict_proba(X_test_scaled)[:, 1]

# Display model evaluation results
st.write("Classification Report:")
st.text(classification_report(y_test, y_pred))
st.write(f"AUC-ROC Score: {roc_auc_score(y_test, y_prob):.2f}")

# Display confusion matrix
st.write("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
st.pyplot()

# Display ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, marker='.', label='Random Forest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
st.write("ROC Curve:")
st.pyplot()

# Streamlit App for Real-Time Prediction
st.sidebar.header('Input Sensor Data')

# Inputs for the sidebar based on the AI4I dataset features
air_temp = st.sidebar.number_input('Air Temperature [K]', min_value=290.0, max_value=320.0, value=300.0)
process_temp = st.sidebar.number_input('Process Temperature [K]', min_value=305.0, max_value=320.0, value=310.0)
rotational_speed = st.sidebar.number_input('Rotational Speed [rpm]', min_value=1000, max_value=3000, value=1500)
torque = st.sidebar.number_input('Torque [Nm]', min_value=0.0, max_value=100.0, value=50.0)
tool_wear = st.sidebar.number_input('Tool Wear [min]', min_value=0, max_value=300, value=150)
machine_type = st.sidebar.selectbox('Machine Type', ['L', 'M', 'H'])

# Encode the 'Machine Type' input similarly to the preprocessing step
type_encoded = {'L': 0, 'M': 1, 'H': 2}[machine_type]

# Create a DataFrame for the input data to ensure feature consistency
input_data = pd.DataFrame([[type_encoded, air_temp, process_temp, rotational_speed, torque, tool_wear]], 
                          columns=X.columns)

# Predict failure probability based on input
input_data_scaled = scaler.transform(input_data)
failure_prob = rf_model.predict_proba(input_data_scaled)[:, 1]

st.write(f"Predicted Probability of Machine Failure: {failure_prob[0]:.2f}")
