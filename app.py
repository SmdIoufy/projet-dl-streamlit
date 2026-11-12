import streamlit as st


st.set_page_config(
    page_title="Projet Deep Learning",
    page_icon="🤖",
    layout="centered"
)


st.title("🤖 Projet Deep Learning")

st.write(
    """
    Bienvenue sur la plateforme de démonstration des modèles
    de Deep Learning développés dans le cadre du Master Data Science
    et Génie Logiciel.
    """
)


st.subheader("Travaux pratiques disponibles")


st.info(
    """
    ### TP1 — Bank Marketing

    Objectif :
    Prédire si un client est susceptible de souscrire
    à un produit bancaire.

    Modèle utilisé :
    - Réseau de neurones profond (DNN)
    - Optimisation des hyperparamètres avec Keras Tuner
    - Déploiement avec Streamlit
    """
)


st.write(
    "Utilisez le menu à gauche pour accéder aux différents TP."
)


st.caption(
    "Plateforme développée progressivement TP par TP."
)