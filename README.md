# fuel_pricing
Create a Virtual Environmen
python -m venv .venv
.venv\Scripts\activate


pip install -r requirements.txt


Train the Model
python src/01_training.py


Get Today’s Optimal Price Recommendation
python src/predict_today.py
