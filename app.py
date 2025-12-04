import streamlit as st
import pickle
import requests
import pandas as pd

MODEL_URL = "https://github.com/cheikhouna033/INCLUSION_FINANCIERE_EN_AFRIQUE/releases/download/Streamlit/model.pkl"

@st.cache_resource
def load_model():
    try:
        r = requests.get(MODEL_URL)
        r.raise_for_status()
        return pickle.loads(r.content)
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

data = load_model()

st.title("Prédiction de l'inclusion financière")

if data is None:
    st.stop()

model = data["model"]
columns = data["columns"]

inputs = {}
for col in columns:
    inputs[col] = st.number_input(f"Valeur pour {col}", value=0)

df_input = pd.DataFrame([inputs])

if st.button("Prédire"):
    pred = model.predict(df_input)[0]
    st.success(f"Résultat : {pred}")
