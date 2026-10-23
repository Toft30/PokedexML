import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import joblib

#laste inn data 
df = pd.read_csv('data/Pokemon.csv')

df['Type 2'] = df['Type 2'].fillna('None')
df['Generation'] = df['Generation'].astype(str)
features = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
type_dummies = pd.get_dummies(df[['Type 1', 'Type 2', 'Generation']])

X = pd.concat([df[features], type_dummies], axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Trener modellen...")
model = NearestNeighbors(n_neighbors=3, metric = 'euclidean')
model.fit(X_scaled)

joblib.dump({
    'model': model,
    'scaler': scaler,
    'features': X.columns,
    'df': df[['Name', 'Type 1', 'Type 2', 'Generation'] + features]
}, 'pokemon_model.pkl')
print("WE done! Modellen er lagra som 'pokemon_model.pkl")


