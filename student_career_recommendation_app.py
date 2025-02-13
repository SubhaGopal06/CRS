
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
@st.cache
def load_data():
    return pd.read_csv('cs_students.csv', encoding='ascii')

df = load_data()

# Train the model (for simplicity, we will retrain it here)
@st.cache
def train_model():
    features = df[['Age', 'GPA']]
    X = features.drop('GPA', axis=1)
    Y = features['GPA']
    model = LinearRegression()
    model.fit(X, Y)
    return model

model = train_model()

# Streamlit app layout
st.title('Student Career Recommendation Dashboard')

# # Login/Signup Page (for simplicity, we will skip actual authentication)
# if st.sidebar.checkbox('Login/Signup'):
#     username = st.sidebar.text_input('Username')
#     password = st.sidebar.text_input('Password', type='password')
#     if st.sidebar.button('Submit'):
#         st.sidebar.success('Logged in successfully!')

# Dropdown menu for student names
student_names = df['Name'].tolist()
selected_student = st.selectbox('Select a Student:', student_names)

# Displaying student details
student_details = df[df['Name'] == selected_student].iloc[0]

st.write(f'**Name:** {student_details["Name"]}')
st.write(f'**Age:** {student_details["Age"]}')
st.write(f'**GPA:** {student_details["GPA"]}')
st.write(f'**Major:** {student_details["Major"]}')
st.write(f'**Interested Domain:** {student_details["Interested Domain"]}')

# Career recommendation (dummy logic for demonstration)
if student_details["GPA"] >= 3.5:
    career_recommendation = "Software Engineer"
else:
    career_recommendation = "Data Analyst"

st.write(f'**Predicted Career Recommendation:** {career_recommendation}')

# Skill graph displayed on the right side
skills = ["Python", "SQL", "Java"]
skill_levels = [3, 2, 1]  # Dummy skill levels

fig, ax = plt.subplots()
ax.barh(skills, skill_levels, color='skyblue')
ax.set_xlabel('Skill Level')
ax.set_title('Skills Overview')

st.pyplot(fig)
