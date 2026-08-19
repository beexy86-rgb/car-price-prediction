# Ovo je skripta za dobijanje preciscenih podataka

import pandas as pd
from src.data_cleaning import clean
from src.feature_engineering import build_features

#Putanje do fajlova u različitim fazama pripreme podataka
RAW_DATA_PATH = "data/cars.csv"
CLEANED_DATA_PATH = "data/car_data_cleaned.csv"
FEATURES_DATA_PATH = "data/car_data_cleaned_with_features.csv"

#Učitavanje sirovih podataka
print("Loading raw data...")
df = pd.read_csv(RAW_DATA_PATH)

#Čišćenje podataka i snimanje u folder data
print("Cleaning data...")
df_cleaned = clean(df)
df_cleaned.to_csv("data/car_data_cleaned.csv")

#Izračunavanje dodatnih karakteristika i snimanje obogaćenog fajla
print("Building features...")
df_featured = build_features(pd.read_csv(CLEANED_DATA_PATH))
print("Saving feature-engineered dataset...")
df_featured.to_csv(FEATURES_DATA_PATH, index=False)
print(f"Feature-engineered dataset saved to: {FEATURES_DATA_PATH}")

#PRAVLJENJE PRVIH PREDIKCIJA - videti da li to raditi u ovom fajlu ili ne



