import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os

MODEL_URL = "https://github.com/cheikhouna033/Inclusion-financiere/releases/download/v1.0/fin_inclusion_model.pkl"

MODEL_PATH = "model.pkl"


# ---------------------------------------------------
# Fonction pour télécharger automatiquement le modèle
# ---------------------------------------------------
def download_model():
    if not os.path.exists(MODEL_PATH):        
        st.info("Téléchargement du modèle en cours...")
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        st.success("Modèle téléchargé avec succès !")


# ---------------------------------------------------
# Charger le modèle
# ---------------------------------------------------
@st.cache_resource
def load_model():
    download_model()
    return joblib.load(MODEL_PATH)


# ---------------------------------------------------
# Interface Streamlit
# ---------------------------------------------------
st.title("Inclusion Financière en Afrique – Prédiction")

st.write("Ce modèle prédit la probabilité d'inclusion financière basée sur vos données.")

model = load_model()

# -----------------------------
# Formulaire utilisateur
# -----------------------------
st.header("Entrer les informations")

age = st.number_input("Âge", min_value=18, max_value=100)
revenu = st.number_input("Revenu mensuel", min_value=0)
education = st.selectbox("Niveau d'éducation", ["Aucun", "Primaire", "Secondaire", "Supérieur"])
sexe = st.selectbox("Sexe", ["Homme", "Femme"])

# Encoder
education_map = {"Aucun": 0, "Primaire": 1, "Secondaire": 2, "Supérieur": 3}
sexe_map = {"Homme": 0, "Femme": 1}

# ---------------------------
# Prédiction
# ---------------------------
if st.button("Prédire"):
    data = np.array([
        age,
        revenu,
        education_map[education],
        sexe_map[sexe]
    ]).reshape(1, -1)

    prediction = model.predict(data)
    proba = model.predict_proba(data)[0][1]

    st.subheader("Résultat")

    if prediction[0] == 1:
        st.success(f"Inclus financièrement (probabilité = {proba:.2f})")
    else:
        st.error(f"Non inclus financièrement (probabilité = {proba:.2f})")
