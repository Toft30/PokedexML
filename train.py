import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

#laste inn data 
df = pd.read_csv('data/Pokemon.csv')

#velge ein stat og kva du vil predikere(type 1)
features = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
X = df[features]
Y = df['Type 1']

#Splitt i trennings og testsett
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#faktisk trene modellen
print("Trener modellen...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)

#evaluere modellen
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(Y_test, y_pred):.2f}")
print("\nKlassifiseringsrapport")
print(classification_report(Y_test, y_pred))

joblib.dump(model, 'pokemon_model.pkl')
print("WE done! Modellen er lagra som 'pokemon_model.pkl")


