
import streamlit as st
import numpy as np
import pandas
import sklearn
import pickle

# Load the trained model and scalers
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))  # Assuming you saved it as 'standscaler.pkl'
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

# Crop dictionary for mapping predictions
crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
             8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
             14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
             19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

# Streamlit app title
st.title("Crop Recommendation System")

# Input fields for features
N = st.number_input("Nitrogen", value=0)
P = st.number_input("Phosphorus", value=0)
K = st.number_input("Potassium", value=0)
temp = st.number_input("Temperature", value=0.0)
humidity = st.number_input("Humidity", value=0.0)
ph = st.number_input("pH", value=0.0)
rainfall = st.number_input("Rainfall", value=0.0)

# Prediction function
def predict_crop(N, P, K, temp, humidity, ph, rainfall):
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)
    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    prediction = model.predict(final_features)
    return prediction[0]

# Predict button
if st.button("Predict"):
    prediction = predict_crop(N, P, K, temp, humidity, ph, rainfall)
    if prediction in crop_dict:
        crop = crop_dict[prediction]
        st.success(f"{crop} is the best crop to be cultivated right there")
    else:
        st.error("Sorry, we could not determine the best crop to be cultivated with the provided data.")