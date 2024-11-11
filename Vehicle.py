import streamlit as st
import numpy as np
import json
from streamlit_option_menu import option_menu
import requests
from streamlit_lottie import st_lottie
from PIL import Image
from tensorflow.keras.models import load_model

def load_image(image_file):
    img = Image.open(image_file)
    img = img.resize((32, 32))
    img = np.array(img)
    if img.shape[-1] == 4:  # Check if the image has an alpha channel
        img = img[..., :3]  # Remove the alpha channel
    img = img.reshape(1, 32, 32, 3)
    img = img.astype('float32')
    img /= 255.0
    return img

# Function to make predictions
def predict(image, model, labels):
    img = load_image(image)
    result = model.predict(img)
    predicted_class = np.argmax(result, axis=1)
    return labels[predicted_class[0]]

# Load the model
model = load_model('vehicle.h5')  

# Function to load labels from a text file
def load_labels(filename):
    with open(filename, 'r') as file:
        labels = file.readlines()
    labels = [label.strip() for label in labels]
    return labels

st.set_page_config(page_title="Vehicle Classification", page_icon=":bus:", layout="wide")

def get(path:str):
    with open(path,"r") as p:
        return json.load(p)

car_path = get("./assets/car.json")
team_path = get("./assets/team.json")
ano_path = get("./assets/Ano.json")
to_path = get("./assets/to.json")
motor_path = get("./assets/motor.json")
truck_path = get("./assets/truckkun.json")

bg_image_path = "./assets/jpeg.jpg"

# Sidebar
with st.sidebar:
    selected = option_menu(
         menu_title = "Main Menu",
        options = ["Home", "About Project", "Vehicle Classification", "Team"],
        icons = ["house", "book", "pin","people"],
        menu_icon ="cast",
        default_index = 0,
    )

#Background Images
st.markdown(
    f"""
    <style>
    .reportview-container {{
        background-image: url('{bg_image_path}');
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Home Page
if selected == "Home":
    st.write("<div style ='text-align: center; font-size: 50px;'> Welcome to Vehicle Classification  </div>", unsafe_allow_html=True)
    with st.container():
        st.write("---")
        left_column, right_column = st.columns(2)
        with left_column:
            st_lottie(car_path, height = 300, key = "hi")
            st_lottie(truck_path, height = 300, key = "hiii")
        with right_column:
            st_lottie(motor_path, height = 300, key = "hiiii")

# ABOUT PROJECT
if selected == "About Project":
    st.header("About Project")

    with st.container():
        st.write("---")
        left_column, right_column = st.columns(2)
        with left_column:
            st.write(
                """
                <div style='text-align: justify;'>
                    Flooding is one of the most frequent and damaging natural disasters in Metro Manila,
                    resulting in significant economic loss, displacement of residents, and disruption of daily life. As
                    climate change and urbanization continue to exacerbate this issue, there is a growing need for
                    effective flood prediction systems that can offer timely warnings and risk assessments. In this
                    project, we propose the development of a flood prediction system using deep learning
                    algorithms, specifically Convolutional Neural Networks (CNN) and Long Short-Term Memory
                    (LSTM) networks, to analyze historical and real-time meteorological data to predict the
                    likelihood of flooding in various parts of Metro Manila.
                    By integrating CNN for spatial pattern recognition and LSTM for temporal sequence
                    analysis, we aim to create a hybrid model that can offer accurate and timely predictions. The
                    system will focus on learning from environmental features such as rainfall, water levels, and
                    geographic data, allowing for efficient flood risk assessment and disaster preparedness. 
                </div>
                """,
                unsafe_allow_html=True
            )
        with right_column:
            st_lottie(ano_path, height = 250, key = "hi")

    with st.container():
        st.write("---")
        left_column, right_column = st.columns(2)
        with right_column:
            st.header("Dataset")
            st.write(
                """
                <div style='text-align: justify;'>
                    The dataset for this project came from this link: <a href='https://www.kaggle.com/datasets/kaggleashwin/vehicle-type-recognition' target='_blank'>Kaggle Vehicle Type Recognition</a>. <br>
                    It contains the following: <br>
                    - Flood  <br>                  
                    - No flood    <be>
                </div>
                """,
                unsafe_allow_html=True
           )
        with left_column:
                st_lottie(to_path, height=250, key="h1")


# Flood Classification
if selected == "Flood Classification":
     st.title("Model Prediction")
     test_image = st.file_uploader("Choose an Image:")
     if test_image is not None:
         st.image(test_image, width=300, caption='Uploaded Image')
         if st.button("Predict"):
             st.write("Predicting...")
             labels = load_labels("labels.txt")
             predicted_sport = predict(test_image, model, labels)
             st.success(f"Predicted Flood Category: {predicted_sport}")

# Team Page
if selected == "Team":
    st.header("Meet the Team")
    with st.container():
        st.write("---")
        left_column, right_column = st.columns(2)
        with left_column:
            st.write(
                """
                Lim, Jhoris T.<br>
                Cadongonan, Nicholo <br>
                Masangkay, Shan Joshua C.<br>
            </div>
            """,
                unsafe_allow_html=True
        )
            st_lottie(team_path, height = 500, key = "hii")
