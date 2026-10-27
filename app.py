
import numpy as np
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="PokédPredict",
    page_icon="⚡",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("pokemon_model.pkl")

data = load_model()
model = data['model']
scaler = data['scaler']
features = data['features']
df = data['df']

st.title("⚡ PokéPredict")
st.write("Skriv inn type og stats, så tipper modellen vår kven det er!")
st.divider()

st.subheader("Elementtypar")
types = sorted([t for t in df['Type 1'].unique() if t != 'None'])

col_t1, col_t2 = st.columns(2)
with col_t1:
    valgt_t1 = st.selectbox("Primærtype (Type 1):", types, index=types.index("Fire") if "Fire" in types else 0)
with col_t2:
    valgt_t2 = st.selectbox("Sekundærtype (Type 2 - valfritt):", ["None"] + types)

generasjoner = sorted(df['Generation'].astype(str).unique())
valgt_gen = st.selectbox("Generasjon: ", generasjoner)

st.subheader("Stats")
col1, col2 = st.columns(2)
with col1:
    hp = st.number_input("HP", 1, 255, 60)
    attack = st.number_input("Attack", 1, 255, 70)
    defense = st.number_input("Defense", 1, 255, 60)
with col2:
    sp_atk = st.number_input("Special Attack", 1, 255, 65)
    sp_def = st.number_input("Special Defense", 1, 255, 65)
    speed = st.number_input("Speed", 1, 255, 60)

st.write("")

if st.button("Finn Pokémon", type="primary", use_container_width=True):
    input_row = {col: 0.0 for col in features}
    
    input_row['HP'] = hp
    input_row['Attack'] = attack
    input_row['Defense'] = defense
    input_row['Sp. Atk'] = sp_atk
    input_row['Sp. Def'] = sp_def
    input_row['Speed'] = speed
    
    t1_col = f"Type 1_{valgt_t1}"
    if t1_col in input_row:
        input_row[t1_col] = 1.0
        
    t2_col = f"Type 2_{valgt_t2}"
    if t2_col in input_row:
        input_row[t2_col] = 1.0

    gen_col = f"Generation_{valgt_gen}"
    if gen_col in input_row:
        input_row[gen_col] = 1.0

    input_df = pd.DataFrame([input_row])
    input_scaled = scaler.transform(input_df)

    distances, indices = model.kneighbors(input_scaled, n_neighbors=3)
    
    inv_dist = 1.0 / (distances[0] + 1e-5)
    similarities = inv_dist / np.sum(inv_dist)

    st.divider()
    topp_treff = df.iloc[indices[0][0]]
    st.success(f"### Mest truleg Pokémon: **{topp_treff['Name']}** ({similarities[0]:.1%})")

    st.subheader("Topp 3 pokemons")
    for i in range(3):
        p_data = df.iloc[indices[0][i]]
        prob = similarities[i]
        
        c1, c2 = st.columns([3, 1])
        c1.write(f"**{i+1}. {p_data['Name']}** ({p_data['Type 1']}" + (f" / {p_data['Type 2']})" if p_data['Type 2'] != 'None' else ")"))
        c2.write(f"{prob:.1%}")
        st.progress(float(prob))
