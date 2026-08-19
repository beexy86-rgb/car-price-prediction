import re
import pandas as pd

RAW_DATA_PATH = "data/car_data_raw.csv"
CLEANED_DATA_PATH = "data/car_data_cleaned.csv"

MISSING_LIKE_VALUES = {"", " ", "nan", "NaN", "NAN", "null", "Null", "NULL", "none", "None", "NONE",}

numeric_columns = ['priceUSD', 'year', 'mileage(kilometers)', 'volume(cm3)']

categorical_columns = ['make', 'model', 'condition', 'fuel_type', 'color', 'transmission', 'drive_unit', 'segment']

#RUKOVANJE EKSTREMNIM VREDNOSTIMA
#Otklanjanje ekstremno niskih cena automobila u koloni priceUSD
def _solve_suspicious_price(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "priceUSD" in df.columns:
        df.loc[df['priceUSD'] < 1000, 'priceUSD'] = 1000
    return df

#Otklanjanje ekstremno malih kilometraža u koloni mileage(kilometre)
def _solve_suspicious_mileage(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "mileage(kilometers)" in df.columns:
        df.loc[df['mileage(kilometers)'] > 1000000, 'mileage(kilometers)'] = 1000000
    return df

#Otklanjanje ekstremno malih zapremina motora u koloni volume(cm3)
def _solve_suspicious_volume(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "volume(cm3)" in df.columns:
        df.loc[df['volume(cm3)'] > 8000, 'mileage(kilometers)'] = 8000
        df.loc[df['volume(cm3)'] < 650, 'mileage(kilometers)'] = 650
    return df


#STANDARDIZACIJA NAZIVA KOLONA
def _standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    new_columns = []
    for col in df.columns:
        clean_col = col.strip().lower()
 
        clean_col = clean_col.replace("(", "_")
        clean_col = clean_col.replace(")", "")
        clean_col = clean_col.replace("-", "_")
        clean_col = clean_col.replace("/", "_")
 
        clean_col = re.sub(r"\s+", "_", clean_col)
        clean_col = re.sub(r"[^a-z0-9_]", "", clean_col)
        clean_col = re.sub(r"_+", "_", clean_col)
        clean_col = clean_col.strip("_")
 
        new_columns.append(clean_col)
    df.columns = new_columns
    return df


#UKLANJANJE VIŠKA RAZMAKA IZ TEKSTUALNIH VREDNOSTI
def _strip_string_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    text_columns = df.select_dtypes(include=["str"]).columns 
    for col in text_columns:
        df[col] = df[col].astype(str).str.strip() 
    return df


#STANDARDIZACIJA NEDOSTAJUĆIH VREDNOSTI
def _replace_missing_like_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.replace(list(MISSING_LIKE_VALUES), pd.NA)
    return df


#KONVERZIJA NUMERIČKIH KOLONA
def _convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


#STANDARDIZACIJA KATEGORIJSKIH KOLONA
def _clean_categorical_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in categorical_columns:
        if col in df.columns:
            df[col] = (df[col].astype("string").str.strip().str.lower())
    return df
    

#POVEZIVANJE U PIPELINE
def clean(df: pd.DataFrame) -> pd.DataFrame:
 
    df_clean = (
        df
        .pipe(_solve_suspicious_price)
        .pipe(_solve_suspicious_mileage)
        .pipe(_solve_suspicious_volume)
        .pipe(_standardize_column_names)
        .pipe(_strip_string_values)
        .pipe(_replace_missing_like_values)
        .pipe(_convert_numeric_columns)
        .pipe(_clean_categorical_values)
        .reset_index(drop=True)
    )
 
    return df_clean