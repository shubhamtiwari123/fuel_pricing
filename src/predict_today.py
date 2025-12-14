

import json
import pandas as pd

from price_optimizer import recommend_price 

hist_df = pd.read_parquet("data/processed.parquet")


with open("data/today_example.json", "r") as f:
    today = json.load(f)


result = recommend_price(today, hist_df)


print("====== DAILY PRICE RECOMMENDATION ======")
print(f"Recommended price : {result['recommended_price']}")
print(f"Expected volume   : {result['expected_volume']}")
print(f"Expected profit   : {result['expected_profit']}")
