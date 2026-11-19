
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="TP4 - Classification CIFAR-10",
    page_icon="🖼️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models" / "tp4_cifar10"

CLASS_NAMES = [
    "Avion",
    "Automobile",
    "Oiseau",
    "Chat",
    "Cerf",
    "Chien",
    "Grenouille",
    "Cheval",
    "Bateau",
    "Camion",
]

CLASS_NAMES_EN = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


# ============================================================
# CHARGEMENT DU MODÈLE
# ============================================================

@st.cache_resource
def load_cifar_model():
    possible_paths = [
    MODEL_DIR / "vgg16_best.keras",
    MODEL_DIR / "vgg16_cifar10.keras",
    MODEL_DIR / "vgg16_cifar10.h5",
    MODEL_DIR / "model_vgg16.keras",
    MODEL_DIR / "model_vgg16.h5",
                        ]

    model_path = next(
        (path for path in possible_paths if path.is_file()),
        None,
    )

    if model_path is None:
        raise FileNotFoundError(
            f"Aucun modèle CIFAR-10 trouvé dans : {MODEL_DIR}\n"
            "Vérifie le nom et l'emplacement du fichier du modèle."
        )

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    return model, model_path


# ============================================================
# PRÉPARATION DE L'IMAGE
# ============================================================

def prepare_image(image: Image.Image) -> np.ndarray:
    """
    Prépare une image pour un modèle VGG16 entraîné
    avec preprocess_input sur des images RGB de CIFAR-10.
    """
    image = image.convert("RGB")
    image = image.resize((32, 32))

    image_array = np.asarray(image, dtype=np.float32)

    # VGG16 attend ici des pixels RGB dans l'intervalle [0, 255].
    image_array = np.expand_dims(image_array, axis=0)

    # Même prétraitement VGG16 que pendant l'entraînement.
    image_array = preprocess_input(image_array)

    return image_array


# ============================================================
# EN-TÊTE
# ============================================================

st.title("🖼️ TP4 — Classification d'images CIFAR-10")
st.markdown(
    """
    Cette application utilise un réseau de neurones convolutif basé sur
    **VGG16** pour reconnaître des objets dans des images.

    Importez une image pour obtenir la classe prédite par le modèle,
    ainsi que les probabilités estimées pour les dix catégories.
    """
)

st.divider()


# ============================================================
# INFORMATIONS SUR LE PROJET
# ============================================================

st.subheader("📌 Présentation du projet")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Nombre de classes", "10")

with col2:
    st.metric("Taille d'entrée", "32 × 32")

with col3:
    st.metric("Architecture", "VGG16")

st.markdown(
    """
    **Classes reconnues :**
    avion, automobile, oiseau, chat, cerf, chien, grenouille,
    cheval, bateau et camion.

    **Principe :** l'image est convertie en RGB, redimensionnée,
    prétraitée pour VGG16, puis transmise au modèle chargé depuis
    le dossier `models/tp4_cifar10/`.
    """
)

st.info(
    "Pour obtenir une prédiction cohérente, le prétraitement de l'image "
    "doit être identique à celui utilisé pendant l'entraînement du modèle."
)

st.divider()


# ============================================================
# CHARGEMENT DU MODÈLE
# ============================================================

try:
    model, model_path = load_cifar_model()

    st.success("Modèle CIFAR-10 chargé avec succès.")
    st.caption(f"Fichier utilisé : {model_path.name}")

except Exception as error:
    st.error("Impossible de charger le modèle CIFAR-10.")
    st.code(str(error))
    st.stop()


# ============================================================
# INTERFACE DE PRÉDICTION
# ============================================================

st.subheader("🔍 Tester une image")

uploaded_file = st.file_uploader(
    "Choisissez une image à classifier",
    type=["jpg", "jpeg", "png", "webp"],
    help="Importez une image contenant un objet à reconnaître.",
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        col_image, col_result = st.columns([1, 1.2], gap="large")

        with col_image:
            st.markdown("#### Image importée")
            st.image(
                image,
                caption="Image originale",
                use_container_width=True,
            )

            st.caption(
                f"Dimensions originales : {image.width} × {image.height}"
            )

        with col_result:
            st.markdown("#### Résultat de la classification")

            with st.spinner("Analyse de l'image en cours..."):
                input_array = prepare_image(image)
                predictions = model.predict(input_array, verbose=0)

            probabilities = np.asarray(predictions[0]).reshape(-1)

            if len(probabilities) != len(CLASS_NAMES):
                st.error(
                    "Le modèle ne renvoie pas 10 sorties. "
                    "Vérifie son architecture et son fichier."
                )
                st.stop()

            # Si le modèle retourne des probabilités softmax.
            if (
                np.all(probabilities >= 0)
                and np.all(probabilities <= 1)
                and np.isclose(probabilities.sum(), 1.0, atol=1e-3)
            ):
                scores = probabilities
            else:
                # Conversion de logits en probabilités si nécessaire.
                scores = tf.nn.softmax(probabilities).numpy()

            predicted_index = int(np.argmax(scores))
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = float(scores[predicted_index]) * 100

            st.markdown("**Classe prédite**")
            st.subheader(f"🎯 {predicted_class}")

            st.metric(
                "Score du modèle",
                f"{confidence:.2f} %",
            )

            st.progress(float(np.clip(confidence / 100, 0, 1)))

            if confidence < 50:
                st.warning(
                    "Le modèle est peu confiant pour cette image. "
                    "Comparez les scores des différentes classes."
                )

        # ----------------------------------------------------
        # PROBABILITÉS PAR CLASSE
        # ----------------------------------------------------

        st.divider()
        st.subheader("📊 Probabilités par classe")

        sorted_indices = np.argsort(scores)[::-1]

        for index in sorted_indices:
            probability = float(scores[index])

            st.write(
                f"**{CLASS_NAMES[index]}** — {probability * 100:.2f} %"
            )

            st.progress(
                float(np.clip(probability, 0, 1))
            )

        # ----------------------------------------------------
        # TOP 3
        # ----------------------------------------------------

        st.divider()
        st.subheader("🏆 Les trois classes les plus probables")

        top_three = sorted_indices[:3]
        top_cols = st.columns(3)

        for position, index in enumerate(top_three):
            with top_cols[position]:
                st.metric(
                    label=f"Rang {position + 1}",
                    value=CLASS_NAMES[index],
                    delta=f"{scores[index] * 100:.2f} %",
                )

        st.caption(
            "Les scores représentent les sorties du modèle, "
            "et ne garantissent pas que la prédiction soit correcte."
        )

    except Exception as error:
        st.error("Une erreur est survenue pendant la prédiction.")
        st.exception(error)

else:
    st.info(
        "Importez une image ci-dessus pour lancer la classification."
    )

st.divider()

st.caption(
    "TP4 — Deep Learning | Classification d'images avec VGG16 et CIFAR-10"
)
