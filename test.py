import pandas as pd

print("Début du script...")
train_df = pd.read_html('train_data.html')[0]
print("train_data.html chargé...")
test_df = pd.read_html('test_data.html')[0]
print("test_data.html chargé...")

# Afficher les premières lignes de chaque dataframe
print("Premières lignes de train_df:")
print(train_df.head())

print("\nPremières lignes de test_df:")
print(test_df.head())