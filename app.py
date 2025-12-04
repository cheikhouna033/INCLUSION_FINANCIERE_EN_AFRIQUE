import streamlit as st
import pandas as pd
import joblib
import requests
import io

MODEL_URL = "https://github.com/cheikhouna033/INCLUSION_FINANCIERE_EN_AFRIQUE/releases/download/stream/model.pkl"

@st.cache_resource
def load_model():
    try:
        response = requests.get(MODEL_URL)
        response.raise_for_status()
        buffer = io.BytesIO(response.content)
        model = joblib.load(buffer)
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

model = load_model()

st.title("Prédiction - Inclusion Financière en Afrique")

st.write("Remplissez les champs pour prédire la possession d’un compte bancaire.")

country = st.selectbox("Country", ["Kenya","Uganda","Tanzania","Rwanda","Burundi"])
year = st.number_input("Year", 2000, 2030, 2016)
location_type = st.selectbox("Location", ["Rural","Urban"])
cellphone_access = st.selectbox("Cellphone access", ["No","Yes"])
household_size = st.number_input("Household size", 1, 50, 4)
age_of_respondent = st.number_input("Age", 10, 120, 30)
gender_of_respondent = st.selectbox("Gender", ["Male","Female"])
relationship_with_head = st.selectbox("Relationship with Head", ["Head of Household","Spouse","Child","Other"])
marital_status = st.selectbox("Marital Status", ["Married","Single","Divorced","Widowed"])
education_level = st.selectbox("Education level", [
    "No formal education","Primary education","Secondary education","Tertiary education"
])
job_type = st.selectbox("Job type", [
    "Self employed","Formally employed Government","Farming and Fishing",
    "Informally employed","Remittance Dependent"
])

input_df = pd.DataFrame([{
    "country": country,
    "year": year,
    "location_type": location_type,
    "cellphone_access": cellphone_access,
    "household_size": household_size,
    "age_of_respondent": age_of_respondent,
    "gender_of_respondent": gender_of_respondent,
    "relationship_with_head": relationship_with_head,
    "marital_status": marital_status,
    "education_level": education_level,
    "job_type": job_type
}])

if st.button("Prédire"):
    if model is None:
        st.error("Modèle non chargé.")
    else:
        pred = model.predict(input_df)[0]
        if pred == 1:
            st.success("✔ Oui, cette personne est susceptible d'avoir un compte bancaire")
        else:
            st.warning("✖ Non, cette personne est peu susceptible d'avoir un compte bancaire")
