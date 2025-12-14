
import pandas as pd
import numpy as np

def read_raw(path="data/oil_retail_history.csv"):
    df = pd.read_csv(path, parse_dates=["date"])
    return df

def validate_and_clean(df):

    df = df.sort_values("date").reset_index(drop=True)
  
    df = df[(df["price"] > 0) & (df["cost"] >= 0) & (df["volume"] >= 0)]
 
    comps = ["comp1_price","comp2_price","comp3_price"]
    df[comps] = df[comps].ffill().bfill()
    return df

def feature_engineer(df):
    df = df.copy()
    df["comp_mean"] = df[["comp1_price","comp2_price","comp3_price"]].mean(axis=1)
    df["price_diff_mean"] = df["price"] - df["comp_mean"]
    df["margin"] = df["price"] - df["cost"]
    df["margin_pct"] = df["margin"] / df["price"]
    
    df["volume_lag1"] = df["volume"].shift(1)
    df["volume_lag7"] = df["volume"].shift(7)
    df["volume_roll7"] = df["volume"].rolling(7, min_periods=1).mean().shift(1)
    df["price_lag1"] = df["price"].shift(1)
  
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)
    df["month"] = df["date"].dt.month
  
    df = df.dropna().reset_index(drop=True)
    return df

def save_processed(df, path="data/processed.parquet"):
    df.to_parquet(path, index=False)

if __name__ == "__main__":
    df = read_raw()
    df = validate_and_clean(df)
    df = feature_engineer(df)
    save_processed(df)
    print("Processed saved to data/processed.parquet")
