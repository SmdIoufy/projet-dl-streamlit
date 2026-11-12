
import os

import joblib as jb
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model


# ============================================================
# 1. Chemins des ressources
# ============================================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT_DIR, "models", "tp1_bank")


# ============================================================
# 2. Chargement des ressources
# ============================================================
@st.cache_resource
def charger_ressources():
    encoders = jb.load(os.path.join(MODEL_DIR, "encoders.joblib"))
    scaler = jb.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    model = load_model(
        os.path.join(MODEL_DIR, "dnn_bank_optimise.h5"),
        compile=False,
    )
    return encoders, scaler, model


encoders, scaler, model = charger_ressources()


# ============================================================
# 3. Colonnes catégorielles
# ============================================================
COLS_CAT = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]


# ============================================================
# 4. Fonction de prédiction
# ============================================================
def prediction_func(age, cats, duration, campaign, pdays, previous):
    codes = {
        nom: encoders[i].transform([cats[nom]])[0]
        for i, nom in enumerate(COLS_CAT)
    }

    x_new = np.array(
        [
            age,
            codes["job"],
            codes["marital"],
            codes["education"],
            codes["default"],
            codes["housing"],
            codes["loan"],
            codes["contact"],
            codes["month"],
            codes["day_of_week"],
            duration,
            campaign,
            pdays,
            previous,
            codes["poutcome"],
        ]
    ).reshape(1, -1)

    x_new = scaler.transform(x_new)

    proba = float(model.predict(x_new, verbose=0)[0][0])
    return proba


# ============================================================
# 5. Interface utilisateur
# ============================================================
st.title("TP1 — Prédiction de souscription bancaire")

st.write(
    "Renseigne les caractéristiques du client pour estimer "
    "la probabilité qu'il souscrive au produit bancaire."
)

st.caption(
    "Modèle de réseau de neurones entraîné et optimisé avec Keras Tuner."
)

st.subheader("Caractéristiques numériques")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Âge",
        min_value=17,
        max_value=100,
        value=40,
    )

    duration = st.number_input(
        "Durée du dernier contact (secondes)",
        min_value=0,
        max_value=5000,
        value=200,
    )

    campaign = st.number_input(
        "Nombre de contacts pendant la campagne",
        min_value=1,
        max_value=60,
        value=1,
    )

with col2:
    pdays = st.number_input(
        "Nombre de jours depuis le contact précédent",
        min_value=0,
        max_value=999,
        value=999,
    )

    previous = st.number_input(
        "Nombre de contacts avant cette campagne",
        min_value=0,
        max_value=20,
        value=0,
    )

st.subheader("Caractéristiques catégorielles")

cats = {}

col1, col2 = st.columns(2)

for i, nom in enumerate(COLS_CAT):
    with col1 if i % 2 == 0 else col2:
        cats[nom] = st.selectbox(
            nom.replace("_", " ").capitalize(),
            list(encoders[i].classes_),
        )

if st.button("Prédire", type="primary"):
    try:
        proba = prediction_func(
            age=age,
            cats=cats,
            duration=duration,
            campaign=campaign,
            pdays=pdays,
            previous=previous,
        )

        st.subheader("Résultat de la prédiction")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Probabilité de souscription", f"{proba:.1%}")

        with col2:
            st.metric("Probabilité de non-souscription", f"{1 - proba:.1%}")

        if proba >= 0.5:
            st.success("Prédiction du modèle : souscription probable.")
        else:
            st.info("Prédiction du modèle : non-souscription probable.")

        st.progress(max(0.0, min(1.0, proba)))

        st.caption(
            "Outil pédagogique : cette prédiction est une estimation "
            "du modèle et ne garantit pas le comportement réel du client."
        )

    except ValueError as erreur:
        st.error(f"Erreur dans les données saisies : {erreur}")
