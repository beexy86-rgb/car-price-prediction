#car_age, mileage_per_year ili engine_volume_liters
import pandas as pd
import datetime

#Izračunavanje starosti automobila u godinama
def _calculate_car_age(df: pd.DataFrame) -> pd.DataFrame: 
    df = df.copy()
    if 'year' in df.columns:
        df['car_age'] = pd.Timestamp.now().year - df['year'] 
    return df

#Izračunavanje prosečne godišnje kilometraže
def _calculate_mileage_per_year(df: pd.DataFrame) -> pd.DataFrame: 
    df = df.copy()
    if 'mileage_kilometers' in df.columns and 'car_age' in df.columns:
        df['mileage_per_year'] = df['mileage_kilometers']/df['car_age'].clip(lower=1) 
    return df

#Računanje indikatora koji ukazuje da je automobil novijeg datuma
def _is_newer_car(df: pd.DataFrame) -> pd.DataFrame: 
    df = df.copy()
    if 'year' in df.columns and 'car_age' in df.columns:
            df['is_newer_car'] = (df['car_age'] < 7).astype(int)
    return df

#Računanje indikatora koji ukazuje da li je automobil "mnogo prešao"
def _is_high_mileage(df: pd.DataFrame) -> pd.DataFrame: 
    df = df.copy()
    if 'mileage_kilometers' in df.columns:
            df['is_high_mileage'] = (df['mileage_kilometers'] > 150000).astype(int)
    return df
