import pickle
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler  # Reapply the scaler here

# Load the trained model
with open('svm_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Recreate the scaler used during training
scaler = StandardScaler()

# Define a function to make predictions
def make_prediction(features):
    # Scale the input features before passing them to the model (re-apply the same scaler)
    features_scaled = scaler.fit_transform([features])  # Scale using the same logic as training
    return model.predict(features_scaled)

# Streamlit app layout
st.title("SVM Model Prediction App")

# User input for features
feature1 = st.number_input("Enter Feature 1", min_value=0.0, max_value=100.0, value=0.0)
feature2 = st.number_input("Enter Feature 2", min_value=0.0, max_value=100.0, value=0.0)

# You can add more input fields for other features if needed

# When the user presses the button, make a prediction
if st.button("Predict"):
    prediction = make_prediction([feature1, feature2])  # Adjust based on the number of features
    st.write(f"The predicted class is: {prediction[0]}")
