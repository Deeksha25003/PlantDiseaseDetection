import streamlit as st
import tensorflow as tf
import numpy as np
import uuid
import os
# CSS code
st.write('<style> \
    /* your CSS code here */ \
    { \
        box-sizing: border-box; \
        margin: 0; \
        padding: 0; \
    } \
    body { \
        font-family: Arial, sans-serif; \
        font-size: 16px; \
        line-height: 1.5; \
        color: #333; /* Changed text color to #333 (a dark gray color) */ \
        background-color: #f9f9f9; \
    } \
    h1, h2, h3, h4, h5, h6 { \
        font-weight: bold; \
        color: #337ab7; \
    } \
    p { \
        margin-bottom: 20px; \
        color: #333; /* Changed text color to #333 (a dark gray color) */ \
    } \
    /* Streamlit Styles */ \
    .stApp { \
        max-width: 1200px; \
        margin: 40px auto; \
        padding: 20px; \
        background-color: #fff; \
        border: 1px solid #ddd; \
        border-radius: 10px; \
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1); \
    } \
    .stSidebar { \
        background-color: #f0f0f0; \
        padding: 20px; \
        border-right: 1px solid #ddd; \
    } \
    .stSidebar .stSelectbox { \
        margin-bottom: 20px; \
    } \
    .stHeader { \
        background-color: #337ab7; \
        color: #fff; \
        padding: 10px; \
        text-align: center; \
    } \
    .stHeader h1 { \
        font-size: 24px; \
        margin-bottom: 10px; \
    } \
    .stImage { \
        max-width: 100%; \
        height: auto; \
        margin: 20px auto; \
        border-radius: 10px; \
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1); \
    } \
</style>', unsafe_allow_html=True)
st.write('<style>button { color: white; background-color: #337ab7; }</style>', unsafe_allow_html=True)
# Tensorflow Model Prediction
def model_prediction(test_image):
    try:
        model = tf.keras.models.load_model("model.h5")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

    try:
        image = tf.keras.preprocessing.image.load_img(test_image, target_size=(256, 256))
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element

# Sidebar
st.sidebar.title("Plant Disease Recognition System")
st.sidebar.markdown("A web application to recognize plant diseases using deep learning")

app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("Welcome to the Plant Disease Recognition System")
    st.markdown("This web application uses deep learning to recognize plant diseases from images.")
    image_path = r"archive (1)\PlantVillage\Potato___Early_blight\0a8a68ee-f587-4dea-beec-79d02e7d3fa4___RS_Early.B 8461.JPG"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Our system uses a convolutional neural network (CNN) to classify images of plant leaves into different disease categories.
    The model is trained on a large dataset of images and can recognize diseases with high accuracy.
    """)

# About Project
elif app_mode == "About":
    st.header("About the Project")
    st.markdown("""
    #### Dataset
    Our dataset consists of images of plant leaves with different diseases.
    The images are collected from various sources and are labeled with the corresponding disease category.
    """)
    st.markdown("""
    #### Model
    Our model is a convolutional neural network (CNN) that is trained on the dataset.
    The model uses a combination of convolutional and pooling layers to extract features from the images.
    The features are then fed into a fully connected layer to produce the output.
    """)
    st.markdown("""
    #### Accuracy
    Our model has an accuracy of over 90% on the test dataset.
    This means that the model can correctly classify images of plant leaves with high accuracy.
    """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    st.markdown("Upload an image of a plant leaf to recognize the disease")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"])
    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image, width=200, use_column_width=True)
        # Predict button
        if st.button("Predict"):
            st.write("Our Prediction")
            # Save the uploaded image to a temporary file
            filename = f"temp_image_{uuid.uuid4()}.jpg"
            with open(filename, "wb") as f:
                f.write(test_image.getbuffer())
            # Pass the temporary file path to the model_prediction function
            result_index = model_prediction(filename)
            # Reading Labels
            class_name = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']
            st.success("Model is Predicting it's a {}".format(class_name[result_index]))
            # Remove the temporary file
            os.remove(filename)