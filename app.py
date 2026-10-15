from flask import Flask, render_template, request
from pathlib import Path
from datetime import datetime
import json
import os
import sqlite3

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# APP CONFIGURATION
app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "restopredict.db"

# DATASET PATH
possible_paths = [
    BASE_DIR / "dataset" / "zomato.csv",
    BASE_DIR / "dataset" / "Zomato.csv",
    BASE_DIR / "Dataset" / "zomato.csv",
    BASE_DIR / "Dataset" / "Zomato.csv",
]

DATASET_PATH = next(
    (path for path in possible_paths if path.is_file()),
    None
)

if DATASET_PATH is None:
    raise FileNotFoundError(
        "Dataset not found. Put zomato.csv inside the dataset folder."
    )

# LOAD DATASET
df = pd.read_csv(DATASET_PATH, encoding="latin1")


def clean_column_name(column):
    return (
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


df.columns = [clean_column_name(c) for c in df.columns]

TARGET = "aggregate_rating"

if TARGET not in df.columns:
    raise ValueError(
        "Aggregate rating column is missing from the dataset."
    )

df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET]).copy()
df = df[df[TARGET].between(0, 5)].copy()

if df.empty:
    raise ValueError("No valid restaurant ratings found.")

# PREPARE FEATURES
excluded_columns = {
    TARGET,
    "restaurant_id",
    "id",
    "restaurant_name",
    "name",
    "address",
    "rating_color",
    "rating_text",
    "locality_verbose",
}

features = [
    c for c in df.columns
    if c not in excluded_columns
]

if not features:
    raise ValueError("No prediction features found.")

X = df[features].copy()
y = df[TARGET].copy()

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = [
    c for c in features
    if c not in numeric_features
]

for column in categorical_features:
    X[column] = X[column].map(
        lambda value: str(value) if pd.notna(value) else None
    )

# PREPROCESSING
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

try:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True
    )
except TypeError:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse=True
    )

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", encoder)
])

transformers = []

if numeric_features:
    transformers.append(
        ("numeric", numeric_pipeline, numeric_features)
    )

if categorical_features:
    transformers.append(
        ("categorical", categorical_pipeline, categorical_features)
    )

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop"
)

# MACHINE LEARNING MODEL
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X, y)

# DATABASE
def get_connection():
    return sqlite3.connect(DATABASE_PATH)


def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                inputs TEXT NOT NULL,
                prediction REAL NOT NULL,
                category TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)


initialize_database()

# HELPER FUNCTIONS
def get_category(rating):
    if rating >= 4.5:
        return "Excellent"
    elif rating >= 3.5:
        return "Good"
    elif rating >= 2.5:
        return "Average"
    else:
        return "Low"


def safe_number(value):
    try:
        number = float(value)
        return number if pd.notna(number) else 0.0
    except (TypeError, ValueError):
        return 0.0


def get_prediction_history():
    with get_connection() as connection:
        rows = connection.execute("""
            SELECT id, inputs, prediction, category, created_at
            FROM prediction_history
            ORDER BY id DESC
        """).fetchall()

    history = []

    for row in rows:
        try:
            saved_inputs = json.loads(row[1])
        except (TypeError, ValueError):
            saved_inputs = {}

        history.append({
            "id": row[0],
            "inputs": saved_inputs,
            "prediction": round(float(row[2]), 2),
            "category": row[3],
            "created_at": row[4]
        })

    return history


def get_page_data(
    prediction=None,
    category=None,
    form_values=None,
    errors=None
):
    if form_values is None:
        form_values = {}

    total_restaurants = len(df)
    average_rating = round(float(df[TARGET].mean()), 2)

    total_votes = 0
    if "votes" in df.columns:
        total_votes = int(
            pd.to_numeric(df["votes"], errors="coerce")
            .fillna(0).sum()
        )

    average_cost = 0
    if "average_cost_for_two" in df.columns:
        average_cost = round(
            safe_number(
                pd.to_numeric(
                    df["average_cost_for_two"],
                    errors="coerce"
                ).mean()
            ),
            2
        )

    excellent = int((df[TARGET] >= 4.5).sum())
    good = int(
        ((df[TARGET] >= 3.5) & (df[TARGET] < 4.5)).sum()
    )
    average = int(
        ((df[TARGET] >= 2.5) & (df[TARGET] < 3.5)).sum()
    )
    low = int((df[TARGET] < 2.5).sum())

    def percentage(count):
        if total_restaurants == 0:
            return 0
        return round(count / total_restaurants * 100, 1)

    history = get_prediction_history()
    history_count = len(history)

    latest_prediction = history[0] if history else None

    if history:
        prediction_average = round(
            sum(item["prediction"] for item in history)
            / history_count,
            2
        )
    else:
        prediction_average = None

    return {
        "features": features,
        "prediction": prediction,
        "category": category,
        "form_values": form_values,
        "errors": errors or [],
        "total_restaurants": total_restaurants,
        "average_rating": average_rating,
        "total_votes": total_votes,
        "average_cost": average_cost,
        "excellent": excellent,
        "good": good,
        "average": average,
        "low": low,
        "excellent_percent": percentage(excellent),
        "good_percent": percentage(good),
        "average_percent": percentage(average),
        "low_percent": percentage(low),
        "history": history,
        "history_count": history_count,
        "latest_prediction": latest_prediction,
        "prediction_average": prediction_average,
    }

# HOME PAGE
@app.route("/")
def home():
    return render_template(
        "index.html",
        **get_page_data()
    )

# PREDICTION
@app.route("/predict", methods=["POST"])
def predict():
    form_values = {
        column: request.form.get(column, "").strip()
        for column in features
    }

    errors = []
    input_row = {}

    for column in features:
        raw_value = form_values[column]

        if raw_value == "":
            errors.append(
                f"Please enter a value for "
                f"{column.replace('_', ' ').title()}."
            )
            continue

        if column in numeric_features:
            try:
                input_row[column] = float(raw_value)
            except ValueError:
                errors.append(
                    f"{column.replace('_', ' ').title()} "
                    "must be a number."
                )
        else:
            input_row[column] = raw_value

    if errors:
        return render_template(
            "index.html",
            **get_page_data(
                form_values=form_values,
                errors=errors
            )
        )

    input_df = pd.DataFrame(
        [input_row],
        columns=features
    )

    for column in numeric_features:
        input_df[column] = pd.to_numeric(
            input_df[column],
            errors="coerce"
        )

    for column in categorical_features:
        input_df[column] = input_df[column].map(
            lambda value: str(value)
            if pd.notna(value) else None
        )

    try:
        predicted_value = float(model.predict(input_df)[0])
        predicted_value = max(0.0, min(5.0, predicted_value))
        predicted_value = round(predicted_value, 2)

        category = get_category(predicted_value)
        created_at = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        with get_connection() as connection:
            connection.execute("""
                INSERT INTO prediction_history
                (inputs, prediction, category, created_at)
                VALUES (?, ?, ?, ?)
            """, (
                json.dumps(form_values),
                predicted_value,
                category,
                created_at
            ))

        return render_template(
            "index.html",
            **get_page_data(
                prediction=predicted_value,
                category=category,
                form_values=form_values
            )
        )

    except Exception:
        app.logger.exception("Prediction failed")

        return render_template(
            "index.html",
            **get_page_data(
                form_values=form_values,
                errors=[
                    "Prediction failed. Check the terminal "
                    "for the error details."
                ]
            )
        )

# RUN APPLICATION
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="127.0.0.1",
        port=port,
        debug=False
    )