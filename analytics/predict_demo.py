"""Reload the saved Titanic pipeline and predict on raw, unprocessed passengers.

Run from inside the analytics folder (after 02_modeling.ipynb has created the file):
    python predict_demo.py
"""
import joblib
import numpy as np
import pandas as pd

pipeline = joblib.load("titanic_best_pipeline.joblib")   # preprocessing + model in one object

new_passengers = pd.DataFrame([
    {"pclass": 1, "sex": "female", "age": 29,     "sibsp": 0, "parch": 0, "fare": 100.0, "embarked": "C"},
    {"pclass": 3, "sex": "male",   "age": 25,     "sibsp": 0, "parch": 0, "fare": 7.25,  "embarked": "S"},
    {"pclass": 2, "sex": "male",   "age": 4,      "sibsp": 1, "parch": 1, "fare": 26.0,  "embarked": "S"},
    {"pclass": 3, "sex": "female", "age": np.nan, "sibsp": 0, "parch": 2, "fare": 15.5,  "embarked": "Q"},
])

new_passengers["P(survived)"] = pipeline.predict_proba(new_passengers)[:, 1].round(3)
new_passengers["prediction"] = np.where(new_passengers["P(survived)"] >= 0.5, "survived", "died")
print(new_passengers.to_string(index=False))
