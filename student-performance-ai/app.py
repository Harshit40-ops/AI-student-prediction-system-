import joblib
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page Configuration
st.set_page_config(
    page_title="AI Student Performance Dashboard",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main {
background-color:#f5f7fa;
}
h1{
color:#2E86C1;
}
</style>
""", unsafe_allow_html=True)

# Load Model and Columns
import os
import joblib

# Current folder path
base_dir = os.path.dirname(__file__)

# Model path
model_path = os.path.join(base_dir, "student_model.pkl")
model = joblib.load(model_path)

# Column path
columns_path = os.path.join(base_dir, "model_columns.pkl")
model_columns = joblib.load(columns_path)
# Load Dataset for Graphs
import os
import pandas as pd

base_dir = os.path.dirname(__file__)
data_path = os.path.join(base_dir, "Student_Performance.csv")

df = pd.read_csv(data_path)

# Title
st.markdown("<h1 style='text-align:center;'>🎓 AI Student Performance Dashboard</h1>", unsafe_allow_html=True)

# Sidebar Inputs
st.sidebar.header("Student Input Features")

study_hours = st.sidebar.slider("Study Hours",0,12)
attendance = st.sidebar.slider("Attendance Percentage",0,100)
math = st.sidebar.slider("Math Score",0,100)
science = st.sidebar.slider("Science Score",0,100)
english = st.sidebar.slider("English Score",0,100)

# Input DataFrame
input_dict = {
    "study_hours":study_hours,
    "attendance_percentage":attendance,
    "math_score":math,
    "science_score":science,
    "english_score":english
}

input_data = pd.DataFrame([input_dict])

# Match training columns
input_data = input_data.reindex(columns=model_columns, fill_value=0)

# Prediction
prediction = model.predict(input_data)

# Prediction Card
st.metric(label="Predicted Student Score", value=f"{prediction[0]:.2f}")

st.write("---")

# Dashboard Layout
col1, col2 = st.columns(2)

# Score Distribution
with col1:
    st.subheader("📊 Score Distribution")

    fig, ax = plt.subplots()
    sns.histplot(df["overall_score"], bins=20, kde=True, ax=ax)
    ax.set_xlabel("Score")
    ax.set_ylabel("Frequency")

    st.pyplot(fig)

# Study Hours vs Score
with col2:
    st.subheader("📉 Study Hours vs Score")

    fig, ax = plt.subplots()
    ax.scatter(df["study_hours"], df["overall_score"])

    ax.set_xlabel("Study Hours")
    ax.set_ylabel("Overall Score")

    st.pyplot(fig)

st.write("---")

# Correlation Heatmap
st.subheader("🔥 Feature Correlation Heatmap")

fig, ax = plt.subplots(figsize=(10,6))

numeric_df = df.select_dtypes(include=['int64','float64'])

sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)

st.pyplot(fig)

st.write("---")

# Prediction Visualization
st.subheader("📈 Prediction Visualization")

predictions = model.predict(df[model_columns])

fig, ax = plt.subplots()

ax.scatter(df["overall_score"], predictions)

ax.set_xlabel("Actual Score")
ax.set_ylabel("Predicted Score")

ax.set_title("Actual vs Predicted Scores")

st.pyplot(fig)
st.subheader("📊 Feature Importance")

importance = model.coef_

imp_df = pd.DataFrame({
    "Feature": model_columns,
    "Importance": importance
})

imp_df = imp_df.sort_values(by="Importance")

fig, ax = plt.subplots(figsize=(8,6))

ax.barh(imp_df["Feature"], imp_df["Importance"])

ax.set_title("Feature Importance")

st.pyplot(fig)
st.subheader("📈 Interactive Study Hours vs Score")
st.markdown("""
<style>

.stApp {
background: linear-gradient(135deg,#667eea,#764ba2);
color:white;
}

h1{
text-align:center;
font-size:45px;
font-weight:bold;
animation: fadeIn 2s;
}

@keyframes fadeIn{
0%{opacity:0;}
100%{opacity:1;}
}

div.stButton > button {
background: linear-gradient(90deg,#ff512f,#dd2476);
color:white;
border:none;
border-radius:10px;
padding:10px 20px;
font-size:18px;
transition:0.3s;
}

div.stButton > button:hover{
transform:scale(1.05);
box-shadow:0px 0px 15px rgba(0,0,0,0.3);
}

.block-container{
padding-top:2rem;
}

.card {
background: rgba(255,255,255,0.1);
padding:20px;
border-radius:15px;
backdrop-filter: blur(10px);
box-shadow:0px 0px 15px rgba(0,0,0,0.2);
margin-bottom:20px;
}

</style>
""", unsafe_allow_html=True)
st.markdown("<h1>🎓 AI Student Performance Dashboard</h1>", unsafe_allow_html=True)

st.write("Predict student performance using Machine Learning 📊")
st.markdown(f"""
<div class="card">
<h2>📊 Predicted Score: {prediction[0]:.2f}</h2>
</div>
""", unsafe_allow_html=True)
import streamlit as st
import requests
from streamlit_lottie import st_lottie
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
lottie_ai = load_lottie("https://assets3.lottiefiles.com/packages/lf20_w51pcehl.json")
col1, col2 = st.columns(2)

with col1:
    st.title("🎓 AI Student Performance Dashboard")
    st.write("Predict student scores using Machine Learning and interactive analytics.")

with col2:
    st_lottie(lottie_ai, height=300)

st.markdown("""
<style>

.stApp {
background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
color:white;
}

div[data-testid="stMetric"]{
background: rgba(255,255,255,0.1);
padding:20px;
border-radius:15px;
backdrop-filter: blur(10px);
box-shadow:0 8px 32px rgba(0,0,0,0.37);
}

button {
border-radius:12px !important;
transition:0.3s;
}

button:hover {
transform:scale(1.05);
}

</style>
""", unsafe_allow_html=True)

st.metric("Predicted Score", prediction)







