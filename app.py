import streamlit as st
import pandas as pd
import joblib
import requests
from io import BytesIO

# -------------------------------
# 🔹 URL du modèle sur GitHub
# -------------------------------
MODEL_URL = "https://drive.google.com/file/d/1eOUOQTZLrU-_ZeDsaOB49oi7vSADuu3y/view?usp=sharing"

@st.cache_resource
def load_model_from_github(url):
    try:
        r = requests.get(url)
        r.raise_for_status()
        model = joblib.load(BytesIO(r.content))
        return model
    except Exception as e:
        st.error(f"Impossible de charger le modèle depuis GitHub : {e}")
        return None

model = load_model_from_github(MODEL_URL)

st.title("Prédiction - Possession d'un compte bancaire")
st.write("Remplis les champs ci-dessous puis clique sur **Prédire**.")

# -------------------------------
# 🔹 Champs d'entrée utilisateur
# -------------------------------
country = st.selectbox("Country", options=["Kenya","Uganda","Tanzania","Rwanda","Burundi"])
year = st.number_input("Year", min_value=2000, max_value=2030, value=2018)
location_type = st.selectbox("Location type", options=["Rural","Urban"])
cellphone_access = st.selectbox("Cellphone access", options=["No","Yes"])
household_size = st.number_input("Household size", min_value=1, max_value=50, value=4)
age_of_respondent = st.number_input("Age of respondent", min_value=10, max_value=120, value=30)
gender_of_respondent = st.selectbox("Gender", options=["Male","Female"])
relationship_with_head = st.selectbox("Relationship with head", options=["Head of Household","Spouse","Child","Other"])
marital_status = st.selectbox("Marital status", options=["Married","Single","Divorced","Widowed"])
education_level = st.selectbox("Education level", options=["No formal education","Primary education","Secondary education","Tertiary education"])
job_type = st.selectbox("Job type", options=["Self employed","Formally employed Government","Farming and Fishing","Informally employed","Remittance Dependent"])

# -------------------------------
# 🔹 Construire DataFrame d'une ligne
# -------------------------------
input_df = pd.DataFrame([{
    "country": country,
    "year": int(year),
    "location_type": location_type,
    "cellphone_access": cellphone_access,
    "household_size": int(household_size),
    "age_of_respondent": int(age_of_respondent),
    "gender_of_respondent": gender_of_respondent,
    "relationship_with_head": relationship_with_head,
    "marital_status": marital_status,
    "education_level": education_level,
    "job_type": job_type
}])

# -------------------------------
# 🔹 Bouton de prédiction
# -------------------------------
if st.button("Prédire"):
    if model is not None:
        pred = model.predict(input_df)[0]
        st.success(
            "Prédiction : **Yes** — la personne est susceptible d'avoir un compte bancaire."
            if pred == 1 else
            "Prédiction : **No** — la personne n'est pas susceptible d'avoir un compte bancaire."
        )
    else:
        st.error("Impossible de faire la prédiction car le modèle n'a pas été chargé.")
