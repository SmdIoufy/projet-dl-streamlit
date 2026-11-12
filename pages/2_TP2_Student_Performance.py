
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model


# =========================================================
# 1. CONFIGURATION DE LA PAGE
# =========================================================

st.set_page_config(
    page_title="TP2 - Student Performance",
    page_icon="🎓",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "tp2_student"


# =========================================================
# 2. CHARGEMENT DES ARTEFACTS
# =========================================================

@st.cache_resource
def load_artifacts():
    model_path = MODEL_DIR / "dnn_student_performance.keras"
    scaler_path = MODEL_DIR / "robust_scaler.joblib"
    encoder_path = MODEL_DIR / "label_encoder.joblib"
    columns_path = MODEL_DIR / "feature_columns.json"

    model = load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)

    with open(columns_path, "r", encoding="utf-8") as file:
        feature_columns = json.load(file)

    return model, scaler, encoder, feature_columns


try:
    model, scaler, encoder, feature_columns = load_artifacts()
except Exception as error:
    st.error(
        "Impossible de charger les fichiers du modèle TP2. "
        "Vérifiez leur présence dans models/tp2_student."
    )
    st.exception(error)
    st.stop()


# =========================================================
# 3. PRESENTATION
# =========================================================

st.title("🎓 TP2 — Student Performance")
st.write(
    "Cette application utilise un réseau de neurones "
    "pour estimer le Performance Index d'un étudiant."
)

st.info(
    "Renseignez les cinq caractéristiques de l'étudiant, "
    "puis cliquez sur le bouton pour obtenir une prédiction."
)


# =========================================================
# 4. FORMULAIRE DE SAISIE
# =========================================================

with st.form("student_prediction_form"):

    st.subheader("Informations sur l'étudiant")

    hours_studied = st.number_input(
        "Heures d'étude",
        min_value=1,
        max_value=9,
        value=5,
        step=1
    )

    previous_scores = st.number_input(
        "Scores précédents",
        min_value=40,
        max_value=100,
        value=75,
        step=1
    )

    extracurricular = st.selectbox(
        "Activités extrascolaires",
        options=["No", "Yes"],
        format_func=lambda value: (
            "Non" if value == "No" else "Oui"
        )
    )

    sleep_hours = st.number_input(
        "Heures de sommeil par jour",
        min_value=4,
        max_value=9,
        value=7,
        step=1
    )

    sample_papers = st.number_input(
        "Nombre de sujets d'examen pratiqués",
        min_value=0,
        max_value=9,
        value=3,
        step=1
    )

    submitted = st.form_submit_button(
        "Prédire le Performance Index",
        use_container_width=True
    )


# =========================================================
# 5. PRETRAITEMENT ET PREDICTION
# =========================================================

if submitted:
    try:
        # Construire les données avec les noms de colonnes
        # attendus par le modèle.
        input_data = {
            "Hours Studied": hours_studied,
            "Previous Scores": previous_scores,
            "Extracurricular Activities": extracurricular,
            "Sleep Hours": sleep_hours,
            "Sample Question Papers Practiced": sample_papers,
        }

        input_df = pd.DataFrame([input_data])

        # Respecter exactement l'ordre des colonnes
        # utilisé lors de l'entraînement.
        input_df = input_df[feature_columns]

        # Encoder la variable catégorielle.
        input_df["Extracurricular Activities"] = (
            encoder.transform(
                input_df["Extracurricular Activities"]
            )
        )

        # Appliquer le scaler déjà ajusté pendant l'entraînement.
        input_scaled = scaler.transform(input_df)

        # Calculer la prédiction.
        prediction = model.predict(
            input_scaled,
            verbose=0
        )

        performance_index = float(prediction[0][0])

        st.success("Prédiction réalisée avec succès.")

        st.metric(
            label="Performance Index prédit",
            value=f"{performance_index:.2f}"
        )

        st.caption(
            "Cette valeur est une estimation produite par le modèle, "
            "et non une garantie du résultat réel de l'étudiant."
        )

        with st.expander("Voir les données utilisées"):
            st.dataframe(
                pd.DataFrame([input_data]),
                use_container_width=True
            )

    except Exception as error:
        st.error(
            "Une erreur est survenue pendant la prédiction."
        )
        st.exception(error)

