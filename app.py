# Import necessary libraries
import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from urllib.parse import quote

# Load the trained models
with open('crop_recommendation_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)
with open('fertilizer.pkl', 'rb') as model1_file:
    loaded1_model = pickle.load(model1_file)

# Load the pre-trained model
model_filename = "PlantDNet.h5"
model_path = "PlantDNet.h5"
model = tf.keras.models.load_model(model_path)

def model_predict(img_path, model):
    img = image.load_img(img_path, grayscale=False, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.array(x, 'float32')
    x /= 255
    preds = model.predict(x)
    return preds

# Streamlit app
st.title("AgriVedha")

# Navigation
page = st.sidebar.selectbox("Select a page:", ["Home", "Disease Prediction", "Crop Recommendation"])

# Home Page
if page == "Home":
    # Center the content in the home page
    st.markdown(
        """
        <style>
        .center {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 90vh;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Centered content
    st.markdown("<h1>Welcome to the AgriVedha App!</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p>This app provides tools for disease prediction and crop recommendation.</p>",
        unsafe_allow_html=True,
    )

    # Display two buttons
    if st.button("Go to Crop Recommendation"):
        page = "Crop Recommendation"

    if st.button("Go to Disease Prediction"):
        page = "Disease Prediction"


# Disease Prediction Page
if page == "Disease Prediction":
    # Disease Prediction Section
    st.header("Disease Prediction")

    # Explanation for users
    st.write("Upload an image of a plant to predict the likelihood of diseases.")

    # File upload directly in the main content
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    # Display the selected image
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    # Button to trigger disease prediction
    if st.button("Predict Disease"):
        if uploaded_file is not None:
            # Make prediction for the uploaded image
            preds = model_predict(uploaded_file, model)
            disease_class = [
                'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight',
                'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight',
                'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
                'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
                'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
            ]
            predicted_class = disease_class[np.argmax(preds[0])]
            confidence_score = np.max(preds[0])

            # Display results in a neat format
            st.subheader("Prediction Results")
            st.write(f"Predicted Disease: {predicted_class}")
        else:
            st.warning("Please upload an image for disease prediction.")



# Crop Recommendation Page
elif page == "Crop Recommendation":
    st.header("Crop Recommendation")
    # Input fields for crop recommendation
    st.subheader("Input Nutrient Values")
    n_value = st.slider("Nitrogen (N) value", 0, 100, 50)
    p_value = st.slider("Phosphorus (P) value", 0, 100, 50)
    k_value = st.slider("Potassium (K) value", 0, 100, 50)
    # Button to trigger crop recommendation prediction
    if st.button("Predict Crop Recommendation"):
        input_data = [n_value, p_value, k_value]
        prediction1 = loaded_model.predict([input_data])[0]
        input_data1 = [n_value, k_value, p_value]
        prediction2 = loaded1_model.predict([input_data1])[0]
    # Display predictions in a table
        st.subheader("Crop Recommendations")
        st.write("The following crops are recommended based on the nutrient values you have provided.")
        st.write(f"Crop Name: {prediction1}")
        st.write(f"Fertilizer Name: {prediction2}")
