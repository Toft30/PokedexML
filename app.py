import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Pokedex Type predikator",
    page_icon="🔴",
    layout= "centered"
)

st.title("🔴 Pokedec ML - Kva e typen?")
st.write("Juster stats til pokemon under, og la modellen tippe kva type dette er")

@st.cache_resource
def load_model():
    return joblib.load('pokemin_model.pkl')

