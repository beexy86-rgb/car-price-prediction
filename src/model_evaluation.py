import joblib
import pandas as pd
 
from sklearn.model_selection import train_test_split
 
from data_preprocessing import (
    split_features_and_target,
    build_preprocessor
)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "data/car_data_cleaned_with_features.csv"
MODEL_PATH = "models/linear_regression_model.joblib"

print("Loading dataset...")
 
df = pd.read_csv(DATA_PATH)

print("Splitting features and target...")
 
X, y = split_features_and_target(df)

print("Splitting data into training and test sets...")
 
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = joblib.load(MODEL_PATH)

#Pravljenje predikcija nad test skupom
print("Making predictions...")
 
y_pred = model.predict(X_test)

#Računanje metrika regresije
print("Calculating regression metrics...")
 
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

metrics = pd.DataFrame({
    "metric": ["MAE", "MSE", "RMSE", "R2"],
    "value": [mae, mse, rmse, r2],
})
 
print("\nRegression metrics:")
print(metrics)

#Kreiranje tabele za analizu predikcija
print("Creating prediction analysis table...")

# Prvo kreiramo DataFrame sa stvarnim i predviđenim cenama automobila
prediction_analysis = pd.DataFrame({
    "actual_price_usd": y_test,
    "predicted_price_usd": y_pred
})
prediction_analysis["error_usd"] = (
    prediction_analysis["actual_price_usd"]
    - prediction_analysis["predicted_price_usd"]
)
prediction_analysis["absolute_error_usd"] = (
    prediction_analysis["error_usd"].abs()
)
print(prediction_analysis)

#ZAKLJUČAK
#Model u proseku greši u proceni cene polovnog automobila za 2092 dolara, što smatram velikom greškom, jer je u celom skupu podataka prosečna cena automobila 7415 dolara.
#Model pravi i velike greške u proceni, na šta ukazuje RMSE.
#R2 od 0,70 je solidan rezultat, što znači da model solidno objašnjava varijacije ciljne promenljive.