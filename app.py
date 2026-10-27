from flask import Flask, render_template, request, jsonify
import os
import pickle
import pandas as pd

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    r2_score,
    mean_squared_error,
    mean_absolute_error
)
from sklearn.neighbors import KNeighborsClassifier


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)


# =========================================================
# PROJECT PATHS
# =========================================================

# Absolute path of the folder containing app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Models folder inside project
MODEL_FOLDER = os.path.join(BASE_DIR, "models")

# Create models folder automatically if missing
os.makedirs(MODEL_FOLDER, exist_ok=True)

print("\n==========================================")
print("ML STUDIO")
print("==========================================")
print("Project Folder:", BASE_DIR)
print("Models Folder :", MODEL_FOLDER)


# =========================================================
# LOAD DATASET
# =========================================================

print("\nLoading Student Performance Dataset...")

student = fetch_ucirepo(id=320)

df = student.data.features.copy()

for column in student.data.targets.columns:
    df[column] = student.data.targets[column]

df = df.dropna().copy()

print("Dataset Loaded Successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# =========================================================
# CREATE CLASSIFICATION TARGETS
# =========================================================

df["Pass_Fail"] = df["G3"].apply(
    lambda grade: "Pass" if grade >= 10 else "Fail"
)


def performance(grade):

    if grade < 10:
        return "Low"

    elif grade < 15:
        return "Medium"

    else:
        return "High"


df["Performance"] = df["G3"].apply(performance)


# =========================================================
# CLASSIFICATION FEATURES
# =========================================================

CLASSIFICATION_FEATURES = [
    "age",
    "Medu",
    "Fedu",
    "studytime",
    "failures",
    "absences",
    "G1",
    "G2"
]


# =========================================================
# REQUIRED MODEL FILES
# =========================================================

REQUIRED_MODELS = [
    "SimpleLinearRegression.pkl",
    "MultipleLinearRegression.pkl",
    "MultipleLinearRegression3Plus.pkl",
    "BinaryLogisticRegression.pkl",
    "MulticlassLogisticRegression.pkl",
    "DecisionTree.pkl",
    "KNN.pkl",
    "NaiveBayes.pkl",
    "LinearSVM.pkl",
    "NonLinearSVM.pkl",
    "RandomForest.pkl"
]


# =========================================================
# CHECK MODEL FILES
# =========================================================

def check_model_files():

    missing_files = []

    for filename in REQUIRED_MODELS:

        path = os.path.join(
            MODEL_FOLDER,
            filename
        )

        if not os.path.exists(path):
            missing_files.append(filename)

    if missing_files:

        print("\n==========================================")
        print("MODEL FILES MISSING")
        print("==========================================")

        print(
            "\nPut the following files inside:\n"
            + MODEL_FOLDER
        )

        for filename in missing_files:
            print("  ❌", filename)

        print("\nCurrent files inside models folder:")

        current_files = os.listdir(MODEL_FOLDER)

        if current_files:

            for filename in current_files:
                print("  ✓", filename)

        else:

            print("  models folder is EMPTY")

        print("\n==========================================")

        raise FileNotFoundError(
            "Required .pkl model files are missing. "
            "Check the models folder shown above."
        )

    print("\nAll model files found! ✓")


check_model_files()


# =========================================================
# LOAD PICKLE
# =========================================================

def load_pickle(filename):

    path = os.path.join(
        MODEL_FOLDER,
        filename
    )

    print("Loading:", filename)

    with open(path, "rb") as file:
        model = pickle.load(file)

    return model


# =========================================================
# LOAD ALL MODELS
# =========================================================

print("\nLoading ML Models...")


simple_model = load_pickle(
    "SimpleLinearRegression.pkl"
)


multiple_model = load_pickle(
    "MultipleLinearRegression.pkl"
)


multiple_more_model = load_pickle(
    "MultipleLinearRegression3Plus.pkl"
)


binary_model = load_pickle(
    "BinaryLogisticRegression.pkl"
)


multiclass_model = load_pickle(
    "MulticlassLogisticRegression.pkl"
)


decision_tree = load_pickle(
    "DecisionTree.pkl"
)


knn_model = load_pickle(
    "KNN.pkl"
)


naive_bayes = load_pickle(
    "NaiveBayes.pkl"
)


linear_svm = load_pickle(
    "LinearSVM.pkl"
)


nonlinear_svm = load_pickle(
    "NonLinearSVM.pkl"
)


random_forest = load_pickle(
    "RandomForest.pkl"
)


print("\nAll ML Models Loaded Successfully! ✓")


# =========================================================
# REGRESSION EVALUATION
# =========================================================

def evaluate_regression(
    model,
    feature_columns
):

    X = df[feature_columns]

    y = df["G3"]


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


    prediction = model.predict(
        X_test
    )


    r2 = r2_score(
        y_test,
        prediction
    )


    mse = mean_squared_error(
        y_test,
        prediction
    )


    mae = mean_absolute_error(
        y_test,
        prediction
    )


    return {

        "r2": round(
            r2,
            3
        ),

        "mse": round(
            mse,
            3
        ),

        "mae": round(
            mae,
            3
        )
    }


# =========================================================
# CLASSIFICATION EVALUATION
# =========================================================

def evaluate_classifier(
    model,
    target="Performance",
    scaling=False
):

    X = df[
        CLASSIFICATION_FEATURES
    ]

    y = df[target]


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    # Apply same scaling used during training
    if scaling:

        scaler = StandardScaler()

        X_train = scaler.fit_transform(
            X_train
        )

        X_test = scaler.transform(
            X_test
        )


    prediction = model.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        prediction
    )


    cm = confusion_matrix(
        y_test,
        prediction
    )


    return {

        "accuracy": round(
            accuracy * 100,
            2
        ),

        "confusion_matrix":
            cm.tolist()
    }


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    # -----------------------------------------------------
    # REGRESSION
    # -----------------------------------------------------

    simple = evaluate_regression(
        simple_model,
        ["G2"]
    )


    multiple = evaluate_regression(
        multiple_model,
        [
            "G1",
            "G2"
        ]
    )


    multiple_more = evaluate_regression(
        multiple_more_model,
        [
            "G1",
            "G2",
            "studytime",
            "failures",
            "absences"
        ]
    )


    # -----------------------------------------------------
    # BINARY LOGISTIC REGRESSION
    # -----------------------------------------------------

    binary = evaluate_classifier(
        binary_model,
        target="Pass_Fail",
        scaling=True
    )


    # -----------------------------------------------------
    # MULTICLASS LOGISTIC REGRESSION
    # -----------------------------------------------------

    multiclass = evaluate_classifier(
        multiclass_model,
        target="Performance",
        scaling=True
    )


    # -----------------------------------------------------
    # DECISION TREE
    # -----------------------------------------------------

    tree = evaluate_classifier(
        decision_tree,
        target="Performance",
        scaling=False
    )


    # -----------------------------------------------------
    # KNN
    # -----------------------------------------------------

    knn = evaluate_classifier(
        knn_model,
        target="Performance",
        scaling=True
    )


    # -----------------------------------------------------
    # NAIVE BAYES
    # -----------------------------------------------------

    bayes = evaluate_classifier(
        naive_bayes,
        target="Performance",
        scaling=False
    )


    # -----------------------------------------------------
    # LINEAR SVM
    # -----------------------------------------------------

    linear = evaluate_classifier(
        linear_svm,
        target="Performance",
        scaling=True
    )


    # -----------------------------------------------------
    # NON-LINEAR SVM
    # -----------------------------------------------------

    nonlinear = evaluate_classifier(
        nonlinear_svm,
        target="Performance",
        scaling=True
    )


    # -----------------------------------------------------
    # RANDOM FOREST
    # -----------------------------------------------------

    forest = evaluate_classifier(
        random_forest,
        target="Performance",
        scaling=False
    )


    # =====================================================
    # MODEL ARENA
    # =====================================================

    arena = {

        "Logistic Regression":
            multiclass["accuracy"],

        "Decision Tree":
            tree["accuracy"],

        "KNN":
            knn["accuracy"],

        "Naive Bayes":
            bayes["accuracy"],

        "Linear SVM":
            linear["accuracy"],

        "RBF SVM":
            nonlinear["accuracy"],

        "Random Forest":
            forest["accuracy"]

    }


    best_model = max(
        arena,
        key=arena.get
    )


    # =====================================================
    # RANDOM FOREST FEATURE IMPORTANCE
    # =====================================================

    importance = []


    if hasattr(
        random_forest,
        "feature_importances_"
    ):

        values = (
            random_forest
            .feature_importances_
        )


        for feature, value in zip(
            CLASSIFICATION_FEATURES,
            values
        ):

            importance.append({

                "feature":
                    feature,

                "value":
                    round(
                        value * 100,
                        2
                    )
            })


        importance = sorted(
            importance,
            key=lambda item:
                item["value"],
            reverse=True
        )


    # =====================================================
    # DATASET PREVIEW
    # =====================================================

    preview_columns = [

        "age",

        "studytime",

        "failures",

        "absences",

        "G1",

        "G2",

        "G3",

        "Pass_Fail",

        "Performance"

    ]


    preview = (

        df[
            preview_columns
        ]

        .head(8)

        .to_dict(
            orient="records"
        )

    )


    # =====================================================
    # SEND DATA TO HTML
    # =====================================================

    return render_template(

        "index.html",

        rows=df.shape[0],

        columns=df.shape[1],

        missing=int(
            df
            .isnull()
            .sum()
            .sum()
        ),

        preview=preview,

        simple=simple,

        multiple=multiple,

        multiple_more=multiple_more,

        binary=binary,

        multiclass=multiclass,

        tree=tree,

        knn=knn,

        bayes=bayes,

        linear=linear,

        nonlinear=nonlinear,

        forest=forest,

        arena=arena,

        best_model=best_model,

        importance=importance

    )


# =========================================================
# INTERACTIVE KNN
# =========================================================

@app.route(
    "/knn",
    methods=["POST"]
)
def change_knn():

    data = request.get_json(
        silent=True
    ) or {}


    try:

        k = int(
            data.get(
                "k",
                5
            )
        )

    except ValueError:

        return jsonify({
            "error":
                "Invalid K value"
        }), 400


    # Keep K between 1 and 15
    if k < 1 or k > 15:

        return jsonify({
            "error":
                "K must be between 1 and 15"
        }), 400


    X = df[
        CLASSIFICATION_FEATURES
    ]

    y = df[
        "Performance"
    ]


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    # KNN needs scaling
    scaler = StandardScaler()


    X_train = scaler.fit_transform(
        X_train
    )


    X_test = scaler.transform(
        X_test
    )


    # Train with selected K
    model = KNeighborsClassifier(
        n_neighbors=k
    )


    model.fit(
        X_train,
        y_train
    )


    prediction = model.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        prediction
    )


    return jsonify({

        "k":
            k,

        "accuracy":
            round(
                accuracy * 100,
                2
            )

    })


# =========================================================
# RUN FLASK
# =========================================================

if __name__ == "__main__":

    print("\n==========================================")
    print("ML STUDIO IS READY")
    print("==========================================")
    print("Open:")
    print("http://127.0.0.1:5000")
    print("==========================================\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )