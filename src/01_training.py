
import pandas as pd
import joblib
import numpy as np

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


df = pd.read_parquet("data/processed.parquet")


split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx].copy()
test  = df.iloc[split_idx:].copy()


features = [
    "price", "cost", "comp_mean", "price_diff_mean", "margin", "margin_pct",
    "volume_lag1", "volume_lag7", "volume_roll7", "price_lag1",
    "day_of_week", "is_weekend", "month"
]

X_train = train[features]
y_train = train["volume"]

X_test = test[features]
y_test = test["volume"]


model = XGBRegressor(
    n_estimators=300,        
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="reg:squarederror"
)

model.fit(X_train, y_train)

joblib.dump(
    {"model": model, "features": features},
    "outputs/xgb_demand_model.joblib"
)


preds = model.predict(X_test)

mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)

mae = mean_absolute_error(y_test, preds)

print(f"RMSE: {rmse:.2f}")
print(f"MAE:  {mae:.2f}")
