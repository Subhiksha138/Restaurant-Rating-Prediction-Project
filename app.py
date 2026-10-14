<<<<<<< HEAD
from flask import Flask, render_template, request
import pandas as pd
import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor


# ---------------- APP CONFIGURATION ----------------

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "Zomato.csv"
DATABASE_PATH = BASE_DIR / "restopredict.db"


# ---------------- LOAD DATASET ----------------

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}. "
        "Place Zomato.csv inside the dataset folder."
    )

df = pd.read_csv(DATASET_PATH, encoding="latin1")


def clean_column_name(column):
    """Convert column names into lowercase snake_case."""
    return (
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


df.columns = [clean_column_name(column) for column in df.columns]

TARGET = "aggregate_rating"

if TARGET not in df.columns:
    raise ValueError(
        "The dataset must contain an 'Aggregate rating' column. "
        f"Available columns: {list(df.columns)}"
    )

df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET]).copy()

# Keep only valid ratings between 0 and 5.
df = df[df[TARGET].between(0, 5)].copy()

if df.empty:
    raise ValueError("No valid restaurant ratings found in the dataset.")


# ---------------- PREPARE FEATURES ----------------

excluded_columns = [
    TARGET,
    "restaurant_id",
    "id",
    "restaurant_name",
    "name",
    "address",
    "rating_color",
    "rating_text",
    "locality_verbose",
]

features = [
    column
    for column in df.columns
    if column not in excluded_columns
]

if not features:
    raise ValueError("No usable prediction features were found.")

X = df[features].copy()
y = df[TARGET].copy()

# Identify numeric and categorical columns.
numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = [
    column for column in features
    if column not in numeric_features
]

# Convert categorical fields to strings so the encoder handles them consistently.
for column in categorical_features:
    X[column] = X[column].where(X[column].notna(), None)
    X[column] = X[column].map(
        lambda value: str(value) if value is not None else None
    )


# ---------------- MACHINE LEARNING MODEL ----------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

# Compatible with older scikit-learn versions as well as newer versions.
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

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", encoder)
    ]
)

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

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

# Train using the available dataset.
model.fit(X, y)


# ---------------- DATABASE ----------------

def get_connection():
    return sqlite3.connect(DATABASE_PATH)


def initialize_database():
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                inputs TEXT NOT NULL,
                prediction REAL NOT NULL,
                category TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


initialize_database()


# ---------------- HELPER FUNCTIONS ----------------

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
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def get_prediction_history():
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, inputs, prediction, category, created_at
            FROM prediction_history
            ORDER BY id DESC
            """
        ).fetchall()

    history = []

    for row in rows:
        try:
            saved_inputs = json.loads(row[1])
        except (TypeError, ValueError):
            saved_inputs = {}

        history.append(
            {
                "id": row[0],
                "inputs": saved_inputs,
                "prediction": round(float(row[2]), 2),
                "category": row[3],
                "created_at": row[4],
            }
        )

    return history


def get_page_data(
    prediction=None,
    category=None,
    form_values=None,
    errors=None
):
    if form_values is None:
        form_values = {}

    # Dataset statistics
    total_restaurants = int(len(df))
    average_rating = round(float(df[TARGET].mean()), 2)

    if "votes" in df.columns:
        total_votes = int(
            pd.to_numeric(df["votes"], errors="coerce")
            .fillna(0).sum()
        )
    else:
        total_votes = 0

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
    else:
        average_cost = 0

    # Dataset rating distribution
    excellent = int((df[TARGET] >= 4.5).sum())
    good = int(((df[TARGET] >= 3.5) & (df[TARGET] < 4.5)).sum())
    average = int(((df[TARGET] >= 2.5) & (df[TARGET] < 3.5)).sum())
    low = int((df[TARGET] < 2.5).sum())

    def percentage(count):
        if total_restaurants == 0:
            return 0
        return round((count / total_restaurants) * 100, 1)

    # Saved prediction history
    history = get_prediction_history()
    history_count = len(history)

    if history:
        latest_prediction = history[0]
        prediction_average = round(
            sum(item["prediction"] for item in history)
            / history_count,
            2
        )
    else:
        latest_prediction = None
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


# ---------------- HOME PAGE ----------------

@app.route("/")
def home():
    page_data = get_page_data()
    return render_template("index.html", **page_data)


# ---------------- PREDICTION ----------------

@app.route("/predict", methods=["POST"])
def predict():
    form_values = {}
    errors = []

    # Keep entered values so the form does not become empty.
    for column in features:
        form_values[column] = request.form.get(
            column, ""
        ).strip()

    input_row = {}

    for column in features:
        raw_value = form_values[column]

        if raw_value == "":
            errors.append(
                f"Please enter a value for {column.replace('_', ' ').title()}."
            )
            continue

        if column in numeric_features:
            try:
                input_row[column] = float(raw_value)
            except ValueError:
                errors.append(
                    f"{column.replace('_', ' ').title()} must be a number."
                )
        else:
            input_row[column] = raw_value

    if errors:
        page_data = get_page_data(
            form_values=form_values,
            errors=errors
        )
        return render_template("index.html", **page_data)

    # Use the same feature order as during model training.
    input_df = pd.DataFrame(
        [input_row],
        columns=features
    )

    # Convert numeric fields and categorical fields consistently.
    for column in numeric_features:
        input_df[column] = pd.to_numeric(
            input_df[column],
            errors="coerce"
        )

    for column in categorical_features:
        input_df[column] = input_df[column].map(
            lambda value: str(value) if pd.notna(value) else None
        )

    try:
        predicted_value = float(model.predict(input_df)[0])

        # Keep rating within the 0-to-5 range.
        predicted_value = max(
            0.0,
            min(5.0, predicted_value)
        )

        predicted_value = round(predicted_value, 2)
        category = get_category(predicted_value)

        # Save prediction and entered inputs.
        created_at = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        with get_connection() as connection:
            connection.execute(
                """
                INSERT INTO prediction_history
                (inputs, prediction, category, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    json.dumps(form_values),
                    predicted_value,
                    category,
                    created_at
                )
            )

        page_data = get_page_data(
            prediction=predicted_value,
            category=category,
            form_values=form_values
        )

        return render_template("index.html", **page_data)

    except Exception as error:
        page_data = get_page_data(
            form_values=form_values,
            errors=[
                "Prediction could not be completed. "
                f"Please check your input values. Details: {error}"
            ]
        )
        return render_template("index.html", **page_data)


# ---------------- RUN APPLICATION ----------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="127.0.0.1",
        port=port,
        debug=False
=======
from flask import Flask, render_template, request
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor


# =====================================================
# FLASK APPLICATION
# =====================================================

app = Flask(__name__)


# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv(
    "dataset/Zomato.csv",
    encoding="latin1"
)


# =====================================================
# CLEAN COLUMN NAMES
# =====================================================

df.columns = df.columns.str.strip().str.lower()


# =====================================================
# TARGET COLUMN
# =====================================================

target = "aggregate rating"

df[target] = pd.to_numeric(
    df[target],
    errors="coerce"
)

df = df.dropna(
    subset=[target]
)


# =====================================================
# REMOVE UNNECESSARY COLUMNS
# =====================================================

remove_columns = [
    "aggregate rating",
    "restaurant id",
    "restaurant name",
    "address",
    "rating color",
    "rating text",
    "locality verbose"
]


features = [
    column
    for column in df.columns
    if column not in remove_columns
]


# =====================================================
# INPUT AND TARGET
# =====================================================

X = df[features]

y = df[target]


# =====================================================
# NUMERIC AND CATEGORICAL COLUMNS
# =====================================================

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()


categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()


# =====================================================
# NUMERIC PREPROCESSING
# =====================================================

numeric_processor = SimpleImputer(
    strategy="median"
)


# =====================================================
# CATEGORICAL PREPROCESSING
# =====================================================

categorical_processor = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )
    )
])


# =====================================================
# COMBINE PREPROCESSING
# =====================================================

preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_processor,
        numeric_features
    ),
    (
        "categorical",
        categorical_processor,
        categorical_features
    )
])


# =====================================================
# MACHINE LEARNING MODEL
# =====================================================

model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "model",
        RandomForestRegressor(
            n_estimators=20,
            random_state=42,
            n_jobs=-1
        )
    )
])


# =====================================================
# TRAIN AND TEST DATA
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# =====================================================
# TRAIN MODEL
# =====================================================

model.fit(
    X_train,
    y_train
)


# =====================================================
# DASHBOARD STATISTICS
# =====================================================

total_restaurants = len(df)


average_rating = round(
    df[target].mean(),
    2
)


# =====================================================
# TOTAL VOTES
# =====================================================

if "votes" in df.columns:

    df["votes"] = pd.to_numeric(
        df["votes"],
        errors="coerce"
    ).fillna(0)

    total_votes = int(
        df["votes"].sum()
    )

else:

    total_votes = 0


# =====================================================
# AVERAGE COST
# =====================================================

if "average cost for two" in df.columns:

    df["average cost for two"] = pd.to_numeric(
        df["average cost for two"],
        errors="coerce"
    )

    average_cost = round(
        df["average cost for two"].mean(),
        2
    )

else:

    average_cost = 0


# =====================================================
# RATING DISTRIBUTION
# =====================================================

excellent = len(
    df[df[target] >= 4.5]
)


good = len(
    df[
        (df[target] >= 3.5) &
        (df[target] < 4.5)
    ]
)


average = len(
    df[
        (df[target] >= 2.5) &
        (df[target] < 3.5)
    ]
)


low = len(
    df[df[target] < 2.5]
)


# =====================================================
# RATING PERCENTAGES
# =====================================================

if total_restaurants > 0:

    excellent_percent = round(
        excellent / total_restaurants * 100,
        2
    )

    good_percent = round(
        good / total_restaurants * 100,
        2
    )

    average_percent = round(
        average / total_restaurants * 100,
        2
    )

    low_percent = round(
        low / total_restaurants * 100,
        2
    )

else:

    excellent_percent = 0
    good_percent = 0
    average_percent = 0
    low_percent = 0


# =====================================================
# RATING CATEGORY
# =====================================================

def get_category(rating):

    if rating >= 4.5:
        return "Excellent"

    elif rating >= 3.5:
        return "Good"

    elif rating >= 2.5:
        return "Average"

    else:
        return "Low"


# =====================================================
# DATA SENT TO HTML
# =====================================================

def get_page_data():

    return {
        "features": features,
        "numeric_features": numeric_features,

        "total_restaurants": total_restaurants,
        "average_rating": average_rating,
        "total_votes": total_votes,
        "average_cost": average_cost,

        "excellent": excellent,
        "good": good,
        "average": average,
        "low": low,

        "excellent_percent": excellent_percent,
        "good_percent": good_percent,
        "average_percent": average_percent,
        "low_percent": low_percent
    }


# =====================================================
# HOME PAGE
# =====================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        **get_page_data()
    )


# =====================================================
# PREDICTION
# =====================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    data = {}


    for column in features:

        value = request.form.get(
            column,
            ""
        ).strip()


        if value == "":

            data[column] = None


        elif column in numeric_features:

            try:

                data[column] = float(value)

            except ValueError:

                data[column] = None


        else:

            data[column] = value


    input_data = pd.DataFrame(
        [data]
    )


    # Make prediction

    prediction = model.predict(
        input_data
    )[0]


    # Keep rating between 0 and 5

    prediction = max(
        0,
        min(5, prediction)
    )


    prediction = round(
        prediction,
        2
    )


    return render_template(
        "index.html",
        **get_page_data(),
        prediction=prediction,
        category=get_category(
            prediction
        )
    )


# =====================================================
# START FLASK APPLICATION
# =====================================================

if __name__ == "__main__":

    app.run(
        debug=True
>>>>>>> 86b41c4d7cdd7abe568331261f6702c62c4591ba
    )