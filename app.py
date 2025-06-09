import streamlit as st
import pickle

# Load saved artifacts
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("Emotion Classifier")
user_input = st.text_area("Enter your sentence:")

if st.button("Predict"):
    input_vec = vectorizer.transform([user_input])
    prediction = model.predict(input_vec)
    st.write(f"Predicted Emotion: {prediction[0].capitalize()}")