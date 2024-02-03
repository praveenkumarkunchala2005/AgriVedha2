# Import necessary libraries
import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from urllib.parse import quote
import joblib
from translate import Translator

# Load the trained models
with open('crop_recommendation_model.pkl', 'rb') as model_file:
    loaded_model = joblib.load(model_file)
with open('fertilizer.pkl', 'rb') as model1_file:
    loaded1_model = joblib.load(model1_file)

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
st.title("AgroTechCrops")

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
    st.markdown("<h1>Welcome to the AgroTechCrops WebApp!</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p>This app provides tools for disease prediction and crop recommendation.</p>",
        unsafe_allow_html=True,
    )


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
    
    languages = {"English": "en","Spanish": "es","French": "fr","Telugu": "te","Hindi": "hi","Tamil": "ta","Kannada": "kn","Malayalam": "ml"}
    selected_language = st.selectbox("Select Language", list(languages.keys()), index=0)
    if st.button("Select Language"):
        language_code = languages[selected_language]
        st.success(f"Selected Language: {selected_language}")
        st.session_state["selected_language"] = language_code 

    # Button to trigger disease prediction
    if st.button("Predict Disease") and st.session_state.get("selected_language"):
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
            result = predicted_class
            cure = ""
            prevention = ""

            if "Pepper__bell___Bacterial_spot" in result:
                cure = "Cure: Apply copper-based fungicides or bactericides early in the disease cycle for bacterial spot on bell peppers."
                prevention = "Prevention: Practice crop rotation, use resistant pepper varieties, and avoid overhead irrigation to minimize bacterial spot impact."
            elif "Potato___Early_blight" in result:
                cure = "Cure: Apply fungicides with chlorothalonil or copper to control early blight on potatoes."
                prevention = "Prevention: Implement crop rotation, use disease-resistant potato varieties, and provide proper spacing to reduce early blight incidence."
            elif "Potato___Late_blight" in result:
                cure = "Cure: There is no cure for viral diseases. Management involves using virus-resistant tomato varieties and controlling whiteflies, which transmit the virus."
                prevention = "Prevention: Implement measures to control whiteflies, such as using sticky traps and insecticides."
            elif "Tomato_Bacterial_spot" in result:
                cure = "Cure: Utilize fungicides containing chlorothalonil or copper to manage late blight on potatoes."
                prevention = "Prevention: Practice good garden hygiene, employ crop rotation, and choose late blight-resistant potato varieties to minimize the risk of infection."
            elif "Tomato_Early_blight" in result:
                cure = "Cure: Use fungicides with chlorothalonil or copper to treat early blight on tomatoes."
                prevention = "Prevention: Implement crop rotation, provide proper spacing for air circulation, and apply mulch to reduce soil splash and minimize early blight development."
            elif "Tomato_Late_blight" in result:
                cure = "Cure:  Apply fungicides containing chlorothalonil or copper to manage late blight on tomatoes."
                prevention = "Prevention:  Practice good garden hygiene, implement crop rotation, and choose tomato varieties resistant to late blight to minimize the risk of infection."
            elif "Tomato_Leaf_Mold" in result:
                cure = "Cure: Use fungicides, such as those containing chlorothalonil, to manage leaf mold on tomatoes."
                prevention = "Prevention: Ensure proper ventilation, avoid overhead watering, and select tomato varieties resistant to leaf mold to reduce the risk of infection."
            elif "Tomato_Septoria_leaf_spot" in result:
                cure = "Cure: Apply fungicides with copper or mancozeb to manage Septoria leaf spot on tomatoes."
                prevention = "Prevention: Practice crop rotation, provide proper spacing for air circulation, and avoid overhead irrigation to minimize the occurrence of Septoria leaf spot."
            elif "Tomato_Spider_mites_Two_spotted_spider_mite" in result:
                cure = "Cure: Miticides can be used to control two-spotted spider mites on tomatoes."
                prevention = "Prevention: Regularly monitor plants, promote natural enemies (predatory mites), and maintain proper humidity levels to discourage two-spotted spider mite infestations."
            elif "Tomato__Target_Spot" in result:
                cure = "Cure: Fungicides containing azoxystrobin or chlorothalonil can be used to manage target spot on tomatoes."
                prevention = "Prevention: Implement crop rotation, maintain good garden hygiene, and choose tomato varieties resistant to target spot to minimize the risk of infection."
            elif "Tomato__Tomato_YellowLeaf__Curl_Virus" in result:
                cure = "Cure: Unfortunately, there is no cure for Tomato Yellow Leaf Curl Virus (TYLCV). Focus on using virus-resistant tomato varieties."
                prevention = "Prevention: Implement rigorous whitefly control measures, employ physical barriers like insect nets, and practice good garden hygiene by removing and destroying infected plants promptly to prevent the spread of TYLCV."
            elif "Tomato__Tomato_mosaic_virus" in result:
                cure = "Cure: There is no cure for viral diseases. Management involves using virus-resistant tomato varieties and controlling whiteflies, which transmit the virus."
                prevention = "Prevention: Implement measures to control whiteflies, such as using sticky traps and insecticides."
            
            a = "Prediction Results"
            b = "Cure and Prevention Information"
            language_code = st.session_state["selected_language"]
            try:
                translator= Translator(to_lang=language_code)
                translation1 = translator.translate(cure)
                translation2 = translator.translate(prevention)
                # Display the cure and prevention information
                c = translator.translate(a)
                d = translator.translate(b)

                st.subheader(c)
                st.write(result)
                st.subheader(d)
                st.write(translation1)
                st.write(translation2)
            except Exception as e:
                st.error(f"Translation error: {e}")
                st.warning("Please try again or choose a different language.")
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

