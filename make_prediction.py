import joblib
import pandas as pd
 
from sklearn.model_selection import train_test_split
 
from src.data_preprocessing import (
    split_features_and_target,
    build_preprocessor
)

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

loaded_model = joblib.load(MODEL_PATH)

sample_X = X_test.sample(10, random_state=42)
sample_y = y_test.loc[sample_X.index]

sample_predictions = loaded_model.predict(sample_X)

prediction_preview = pd.DataFrame({
    "actual_price_usd": sample_y.values,
    "predicted_price_usd": sample_predictions,
    "error": (sample_predictions - sample_y.values),
})
 
print(prediction_preview)
