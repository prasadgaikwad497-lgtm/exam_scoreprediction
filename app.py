import streamlit as st
import pickle
import numpy as np
import pandas as pd
import time

# Page Configuration
st.set_page_config(
    page_title="Academic Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling & Animations
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        width: 100%;
        background-color: #2563EB;
        color: white;
        font-weight: bold;
        padding: 0.6rem 1rem;
        border-radius: 8px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
        transform: translateY(-2px);
    }
    </style>
""", unsafe_allow_html=True)

# Load Trained Model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model.pkl: {e}")
    st.stop()

# Header
st.markdown('<div class="main-title">🎓 Student Performance Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Enter academic and lifestyle factors below to predict the expected score.</div>', unsafe_allow_html=True)

# Form Inputs Layout
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Demographic & General")
        age = st.number_input("Age", min_value=10, max_value=100, value=20)
        gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        course = st.selectbox("Course Code", options=[0, 1, 2, 3, 4], help="Categorical encoding for enrolled course")

    with col2:
        st.subheader("📚 Study Habits")
        study_hours = st.slider("Weekly Study Hours", 0.0, 60.0, 15.0, 0.5)
        class_attendance = st.slider("Class Attendance (%)", 0.0, 100.0, 85.0, 1.0)
        study_method = st.selectbox("Study Method", options=[0, 1, 2], help="Encoded study methodology")

    with col3:
        st.subheader("🌙 Health & Environment")
        sleep_hours = st.slider("Sleep Hours / Night", 0.0, 12.0, 7.0, 0.5)
        sleep_quality = st.select_slider("Sleep Quality Rating", options=[1, 2, 3, 4, 5], value=3)
        internet_access = st.selectbox("Internet Access", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        facility_rating = st.select_slider("Facility Rating", options=[1, 2, 3, 4, 5], value=3)
        exam_difficulty = st.select_slider("Exam Difficulty", options=[1, 2, 3, 4, 5], value=3)

    st.markdown("<br>", unsafe_allow_html=True)
    submit = st.form_submit_button("✨ Predict Score")

# Prediction Trigger & UI Effects
if submit:
    # Processing Animation Effect
    with st.spinner("Analyzing parameters and running model prediction..."):
        time.sleep(0.8)

    # Feature array aligned with model's expected inputs
    input_data = pd.DataFrame([{
        'age': age,
        'gender': gender,
        'course': course,
        'study_hours': study_hours,
        'class_attendance': class_attendance,
        'internet_access': internet_access,
        'sleep_hours': sleep_hours,
        'sleep_quality': sleep_quality,
        'study_method': study_method,
        'facility_rating': facility_rating,
        'exam_difficulty': exam_difficulty
    }])

    prediction = model.predict(input_data)[0]

    # Visual Celebration Effect
    st.balloons()

    # Display Result
    st.markdown("---")
    res_col1, res_col2, res_col3 = st.columns([1, 2, 1])
    with res_col2:
        st.success("Prediction Complete!")
        st.metric(
            label="Predicted Score / Output",
            value=f"{prediction:.2f}"
        )
