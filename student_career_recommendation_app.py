import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore

# Load the dataset
@st.cache_data  # Updated caching
def load_data():
    return pd.read_csv('cs_students.csv', encoding='ascii')

df = load_data()

# Train the model (for simplicity, we will retrain it here)
@st.cache_resource  # Updated caching
def train_model():
    features = df[['Age', 'GPA']]
    X = features.drop('GPA', axis=1)
    Y = features['GPA']
    model = LinearRegression()
    model.fit(X, Y)
    return model

model = train_model()

# Streamlit app layout
st.title('Smart Career Guidance Dashboard')

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

# Career recommendation logic
career_recommendation = "Software Engineer" if student_details["GPA"] >= 3.5 else "Data Analyst"
st.write(f'**Predicted Career Recommendation:** {career_recommendation}')

# Fetch student-specific skill levels
skills = ["Python", "SQL", "Java"]
skill_levels = [student_details["Python"], student_details["SQL"], student_details["Java"]]

# Skill graph
fig, ax = plt.subplots()
ax.barh(skills, skill_levels, color='skyblue')
ax.set_xlabel('Skill Level')
ax.set_title('Skills Overview')

st.pyplot(fig)
