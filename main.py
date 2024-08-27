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

# Load and preview the dataset
df = pd.read_csv('sensor_data.csv')
st.write("Dataset Preview:")
st.write(df.head())

# Data Preprocessing
df.fillna(method='ffill', inplace=True)
df['temp_roll_mean'] = df['temperature'].rolling(window=5).mean()
df['vibration_roll_std'] = df['vibration'].rolling(window=5).std()
df.dropna(inplace=True)
df['failure'] = (df['maintenance_event'] == 'failure').astype(int)

# Define features and target
X = df.drop(['maintenance_event', 'failure'], axis=1)
y = df['failure']

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
temperature = st.sidebar.number_input('Temperature', min_value=0.0, max_value=100.0, value=50.0)
vibration = st.sidebar.number_input('Vibration', min_value=0.0, max_value=100.0, value=50.0)

# Predict failure probability based on input
input_data = np.array([[temperature, vibration]])
input_data_scaled = scaler.transform(input_data)
failure_prob = rf_model.predict_proba(input_data_scaled)[:, 1]

st.write(f"Predicted Probability of Failure: {failure_prob[0]:.2f}")
