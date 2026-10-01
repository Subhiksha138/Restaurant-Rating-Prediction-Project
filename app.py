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
    )