
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="TP5 - Détection de pneumonie",
    page_icon="🫁",
    layout="wide",
)

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "tp5_pneumonie"
    / "vgg16_optimise_pneumonie.keras"
)

IMAGE_SIZE = (150, 150)
THRESHOLD = 0.5


# ============================================================
# CHARGEMENT DU MODÈLE
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )


# ============================================================
# PRÉTRAITEMENT
# ============================================================

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Le notebook charge les radiographies en niveaux de gris,
    les redimensionne à 150 x 150 puis répète le canal gris
    trois fois pour obtenir une entrée à 3 canaux.

    Aucune normalisation par 255 n'est appliquée ici.
    """

    # Corriger l'orientation éventuelle de la photo
    image = ImageOps.exif_transpose(image)

    # Reproduire le traitement du notebook :
    # niveaux de gris, puis redimensionnement
    image = image.convert("L")
    image = image.resize(IMAGE_SIZE)

    # Tableau de forme (150, 150), valeurs entre 0 et 255
    image_array = np.asarray(image, dtype=np.float32)

    # Répéter le canal gris trois fois : (150, 150, 3)
    image_array = np.repeat(
        image_array[..., np.newaxis],
        3,
        axis=-1,
    )

    # Ajouter la dimension batch : (1, 150, 150, 3)
    return np.expand_dims(image_array, axis=0)


# ============================================================
# INTERFACE
# ============================================================

st.title("🫁 TP5 — Détection de pneumonie avec VGG16")

st.write(
    """
    Cette page utilise le modèle VGG16 optimisé entraîné dans
    le notebook de détection de pneumonie à partir de
    radiographies thoraciques.
    """
)

st.info(
    "Outil pédagogique expérimental : cette prédiction "
    "ne constitue pas un diagnostic médical."
)

with st.expander("À propos du modèle"):
    st.markdown(
        """
        - Architecture : VGG16 optimisé
        - Taille d'entrée : 150 × 150 pixels
        - Canaux : 3, obtenus en répétant le canal gris
        - Sortie : probabilité sigmoid
        - Seuil de classification : 0,5
        """
    )


# ============================================================
# CHARGER LE MODÈLE
# ============================================================

try:
    model = load_model()
except Exception as exc:
    st.error(f"Impossible de charger le modèle : {exc}")
    st.stop()


# Vérification de compatibilité
if model.input_shape[1:] != (150, 150, 3):
    st.error(
        "La forme d'entrée du modèle est incompatible avec "
        f"le prétraitement prévu : {model.input_shape}"
    )
    st.stop()

if model.output_shape[-1] != 1:
    st.error(
        "Le modèle chargé ne possède pas une sortie binaire "
        f"attendue : {model.output_shape}"
    )
    st.stop()


# ============================================================
# TÉLÉVERSEMENT DE L'IMAGE
# ============================================================

uploaded_file = st.file_uploader(
    "Téléverse une radiographie thoracique",
    type=["png", "jpg", "jpeg"],
    help="Formats acceptés : PNG, JPG et JPEG.",
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        image.load()

        col_image, col_result = st.columns(2)

        with col_image:
            st.subheader("Radiographie téléversée")
            st.image(
                image,
                caption="Image originale",
                use_container_width=True,
            )

        with col_result:
            st.subheader("Analyse du modèle")

            if st.button(
                "Analyser la radiographie",
                type="primary",
                use_container_width=True,
            ):
                input_array = preprocess_image(image)

                with st.spinner("Prédiction en cours..."):
                    prediction = model.predict(
                        input_array,
                        verbose=0,
                    )

                score = float(prediction[0][0])

                if not np.isfinite(score) or not 0 <= score <= 1:
                    st.error(
                        "Le modèle a produit une valeur "
                        "invalide. Vérifie le fichier du modèle."
                    )
                    st.stop()

                # Le notebook définit 0 = Normal et 1 = Pneumonia
                if score > THRESHOLD:
                    label = "PNEUMONIA"
                    confidence = score
                    st.error(f"Résultat du modèle : {label}")
                else:
                    label = "NORMAL"
                    confidence = 1.0 - score
                    st.success(f"Résultat du modèle : {label}")

                st.metric(
                    "Score associé à la classe prédite",
                    f"{confidence * 100:.2f} %",
                )

                st.progress(confidence)

                st.caption(
                    f"Score sigmoid brut : {score:.4f} | "
                    f"Seuil utilisé : {THRESHOLD:.2f}"
                )

                st.warning(
                    "Le score est une sortie du modèle, "
                    "pas une probabilité médicale calibrée. "
                    "Cette prédiction ne remplace pas l'avis "
                    "d'un professionnel de santé."
                )

    except Exception as exc:
        st.error(
            "Impossible de lire ou d'analyser cette image. "
            f"Détail : {exc}"
        )
else:
    st.caption(
        "Téléverse une image pour lancer une prédiction."
    )
