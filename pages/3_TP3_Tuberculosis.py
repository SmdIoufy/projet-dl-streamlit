from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# --------------------------------------------------
# Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="TP3 - Détection de la tuberculose",
    page_icon="🫁",
    layout="centered",
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "tp3_tuberculosis" / "vgg16_model.h5"

IMG_SIZE = (150, 150)
THRESHOLD = 0.5


# --------------------------------------------------
# Chargement du modèle
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False,
    )


# --------------------------------------------------
# Prétraitement de l'image
# --------------------------------------------------

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Prépare l'image comme dans le notebook d'entraînement."""

    image = image.convert("RGB")
    image = image.resize(
        IMG_SIZE,
        resample=Image.Resampling.NEAREST,
    )

    image_array = np.asarray(image, dtype=np.float32)
    image_array = image_array / 255.0

    # Forme finale : (1, 150, 150, 3)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# --------------------------------------------------
# Interface utilisateur
# --------------------------------------------------

st.title("🫁 Détection de la tuberculose")
st.caption("TP3 — Deep Learning : classification de radiographies thoraciques")

st.write(
    "Téléverse une radiographie thoracique pour obtenir "
    "une prédiction du modèle entraîné."
)

st.info(
    "Cette application est un prototype pédagogique. "
    "Elle ne remplace pas un diagnostic médical."
)

uploaded_file = st.file_uploader(
    "Choisir une radiographie",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Radiographie téléversée",
            width="stretch",
        )

        if st.button("Analyser la radiographie", type="primary"):
            with st.spinner("Analyse en cours..."):
                model = load_model()
                image_array = preprocess_image(image)

                prediction = model.predict(
                    image_array,
                    verbose=0,
                )

                probability_tb = float(prediction[0][0])

            if not 0.0 <= probability_tb <= 1.0:
                st.error("Le modèle a renvoyé une valeur inattendue.")
                st.stop()

            probability_normal = 1.0 - probability_tb

            if probability_tb >= THRESHOLD:
                predicted_class = "Tuberculosis"
                confidence = probability_tb
            else:
                predicted_class = "Normal"
                confidence = probability_normal

            st.subheader("Résultat de la prédiction")

            if predicted_class == "Tuberculosis":
                st.warning(
                    "Classe prédite : Tuberculosis"
                )
            else:
                st.success(
                    "Classe prédite : Normal"
                )

            st.metric(
                "Classe prédite",
                predicted_class,
            )

            st.write("Scores de sortie du modèle :")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Normal",
                    f"{probability_normal * 100:.2f} %",
                )

            with col2:
                st.metric(
                    "Tuberculosis",
                    f"{probability_tb * 100:.2f} %",
                )

            st.caption(
                f"Score associé à la classe prédite : "
                f"{confidence * 100:.2f} %."
            )

            st.caption(
                "Le seuil de 0,5 et l'interprétation des classes "
                "reprennent le notebook d'entraînement. "
                "Ces scores ne constituent pas des probabilités "
                "cliniquement validées."
            )

    except Exception as error:
        st.error(
            "Une erreur est survenue pendant le chargement "
            "de l'image ou la prédiction."
        )
        st.exception(error)
