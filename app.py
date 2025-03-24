
import streamlit as st
import numpy as np
import joblib
import cv2 #for image processing
from streamlit_drawable_canvas import st_canvas
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import emoji

#Model name with score
model_file = {"SVM(97.18)": "best_model_svm.pkl"}

#Load model
best_model_svm = joblib.load('best_model_svm.pkl')



# Streamlit UI for input
# Main Heading
st.markdown("<h1 style='text-align: center; color: #ff5733;'>📊 MNIST DATA</h1>", unsafe_allow_html=True)

# Subheading
st.markdown("<h2 style='text-align: center;font-style: italic; color: #1f77b4;'>✍️ Handwritten Digit Prediction</h2>", unsafe_allow_html=True)

st.markdown("MNIST-datasetet består av 70 000 bilder av handskrivna siffror, från 0 till 9. Vi tränade flera maskininlärningsmodeller med denna data och valde SVM-modellen baserat på dess noggrannhet på 97,18%.")



# Italicized Description
st.markdown("<p style='text-align: center; font-style: italic;'>This app allows you to draw a digit, and our trained model will predict it using machine learning.</p>", unsafe_allow_html=True)

#canvas image properties
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=20,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict"):
    if canvas_result.image_data is not None: np.any(canvas_result.image_data)
     # Get image from the canvas
    img_array = np.array(canvas_result.image_data)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)  # Convert to grayscale
    img_array = cv2.resize(img_array, (28, 28))  # Resize to 28x28
    img_array = img_array / 255.0  # Normalize
    img_array = img_array.flatten().reshape(1, -1)  # Flatten and reshape


     # Make prediction using the best model
    prediction = best_model_svm.predict(img_array)
    predicted_number = prediction[0]

    # Display the result
    st.markdown(emoji.emojize(f"**Predicted number**: {predicted_number} :thumbs_up:"))
else:
    #Message if the canvas is empty
    st.warning("✍️ please draw a number to predict.")
    


