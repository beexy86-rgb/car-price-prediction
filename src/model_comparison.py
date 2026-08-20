import pandas as pd
 
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
 
from data_preprocessing import (
    split_features_and_target,
    build_preprocessor,
)

#Definisanje putanje i učitavanje podataka sa dodatnim karakteristikama
DATA_PATH = "data/car_data_cleaned_with_features.csv"
print("Loading data...")
df = pd.read_csv(DATA_PATH)

#Razdvajanje podataka i podela na trening i test skup
print("Splitting data...")
X, y = split_features_and_target(df)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#Definisanje algoritama koje želimo da uporedimo
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}

#Treniranje i evaluacija modela
print("Training models...")
results = []
for model_name, regressor in models.items():
 
    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", regressor),
        ]
    )
 
    model.fit(X_train, y_train)
 
    y_pred = model.predict(X_test)
 
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
 
    results.append({
        "model": model_name,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    })
    
#Prikaz i analiza rezultata
results_df = pd.DataFrame(results)

#Sortiranje prema MAE
results_df = results_df.sort_values(
    by="mae",
    ascending=True
)
print(results_df)

