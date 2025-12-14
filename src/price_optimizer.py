

import numpy as np
import pandas as pd
import joblib


bundle = joblib.load("outputs/xgb_demand_model.joblib")
model = bundle["model"]
features = bundle["features"]


MAX_DAILY_PRICE_CHANGE_PCT = 0.05   
MIN_MARGIN_PCT = 0.02              


def recommend_price(today: dict, hist_df: pd.DataFrame):
  

    last_row = hist_df.iloc[-1]

  
    last_price = today["price"]
    cost = today["cost"]

    # Candidate prices
    price_min = last_price * (1 - MAX_DAILY_PRICE_CHANGE_PCT)
    price_max = last_price * (1 + MAX_DAILY_PRICE_CHANGE_PCT)
    candidate_prices = np.linspace(price_min, price_max, 25)

    rows = []

    for p in candidate_prices:
        margin = p - cost
        margin_pct = margin / p

        if margin_pct < MIN_MARGIN_PCT:
            continue

        row = {
            "price": p,
            "cost": cost,
            "comp_mean": np.mean([
                today["comp1_price"],
                today["comp2_price"],
                today["comp3_price"],
            ]),
            "price_diff_mean": p - np.mean([
                today["comp1_price"],
                today["comp2_price"],
                today["comp3_price"],
            ]),
            "margin": margin,
            "margin_pct": margin_pct,
            "volume_lag1": last_row["volume"],
            "volume_lag7": last_row["volume_lag7"],
            "volume_roll7": last_row["volume_roll7"],
            "price_lag1": last_row["price"],
            "day_of_week": pd.to_datetime(today["date"]).dayofweek,
            "is_weekend": int(pd.to_datetime(today["date"]).dayofweek >= 5),
            "month": pd.to_datetime(today["date"]).month,
        }

        X = pd.DataFrame([row])[features]
        pred_volume = model.predict(X)[0]
        profit = margin * pred_volume

        row["pred_volume"] = pred_volume
        row["profit"] = profit

        rows.append(row)

    result = pd.DataFrame(rows)

    best = result.sort_values("profit", ascending=False).iloc[0]

    return {
        "recommended_price": round(best["price"], 2),
        "expected_volume": round(best["pred_volume"], 0),
        "expected_profit": round(best["profit"], 2),
    }
