
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Pokédex AI",
    page_icon="⚡",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("pokemon_model.pkl")

model = load_model()

st.title("⚡ Pokédex AI")
st.write("Predict a Pokémon's primary type using its stats.")

st.subheader("Pokémon stats")

hp = st.number_input("HP", 1, 255, 60)
attack = st.number_input("Attack", 1, 255, 70)
defense = st.number_input("Defense", 1, 255, 60)
sp_atk = st.number_input("Special Attack", 1, 255, 65)
sp_def = st.number_input("Special Defense", 1, 255, 65)
speed = st.number_input("Speed", 1, 255, 60)

if st.button("Predict Pokémon Type", type="primary"):
    pokemon = pd.DataFrame([{
        "HP": hp,
        "Attack": attack,
        "Defense": defense,
        "Sp. Atk": sp_atk,
        "Sp. Def": sp_def,
        "Speed": speed
    }])

    prediction = model.predict(pokemon)[0]
    probabilities = model.predict_proba(pokemon)[0]

    st.success(f"Predicted type: {prediction}")

    results = pd.DataFrame({
        "Type": model.classes_,
        "Probability": probabilities
    }).sort_values("Probability", ascending=False)

    st.subheader("Top 3 predicted types")

    for _, result in results.head(3).iterrows():
        st.write(
            f'{result["Type"]}: '
            f'{result["Probability"]:.1%}'
        )
