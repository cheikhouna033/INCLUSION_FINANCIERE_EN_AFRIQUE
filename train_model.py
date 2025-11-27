import streamlit as st
import pandas as pd
import joblib

MODEL_PATH = "fin_inclusion_model.pkl"  # Le modèle doit être dans le même dossier que app.py

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("Prédiction - Possession d'un compte bancaire")
st.write("Remplis les champs ci-dessous puis clique sur **Prédire**.")

country = st.selectbox("Country", ["Kenya","Uganda","Tanzania","Rwanda","Burundi"])
year = st.number_input("Year", 2000, 2030, 2018)
location_type = st.selectbox("Location type", ["Rural","Urban"])
cellphone_access = st.selectbox("Cellphone access", ["No","Yes"])
household_size = st.number_input("Household size", 1, 50, 4)
age = st.number_input("Age of respondent", 10, 120, 30)
gender = st.selectbox("Gender", ["Male","Female"])
relationship = st.selectbox("Relationship with head", ["Head of Household","Spouse","Child","Other"])
marital = st.selectbox("Marital status", ["Married","Single","Divorced","Widowed"])
education = st.selectbox("Education level", [
    "No formal education","Primary education","Secondary education","Tertiary education"
])
job = st.selectbox("Job type", [
    "Self employed","Formally employed Government","Farming and Fishing",
    "Informally employed","Remittance Dependent"
])

input_df = pd.DataFrame([{
    "country": country,
    "year": year,
    "location_type": location_type,
    "cellphone_access": cellphone_access,
    "household_size": household_size,
    "age_of_respondent": age,
    "gender_of_respondent": gender,
    "relationship_with_head": relationship,
    "marital_status": marital,
    "education_level": education,
    "job_type": job
}])

if st.button("Prédire"):
    pred = model.predict(input_df)[0]
    st.success("Prediction: **Yes**" if pred==1 else "Prediction: **No**")
