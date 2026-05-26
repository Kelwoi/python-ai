# fuel_nn_poly_solution.py
# Task 1: Polynomial regression + neural network comparison for fuel consumption.
# Task 2: Add trip time and engine type as extra input parameters.

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import LeaveOneOut
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore", category=ConvergenceWarning)

CSV_FILE = "fuel_consumption_vs_speed.csv"


class TargetScalerRegressor(BaseEstimator):
    """A small wrapper that scales y for models like MLPRegressor."""
    def __init__(self, regressor=None):
        self.regressor = regressor
        self.y_scaler = StandardScaler()

    def fit(self, X, y):
        y = np.asarray(y).reshape(-1, 1)
        self.y_scaler_ = StandardScaler()
        y_scaled = self.y_scaler_.fit_transform(y).ravel()
        self.regressor_ = clone(self.regressor)
        self.regressor_.fit(X, y_scaled)
        return self

    def predict(self, X):
        y_scaled = self.regressor_.predict(X).reshape(-1, 1)
        return self.y_scaler_.inverse_transform(y_scaled).ravel()


def make_one_hot_encoder():
    """Keeps code compatible with older and newer scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_original_data():
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(
            f"File '{CSV_FILE}' was not found. Put this script in the same folder as the CSV file."
        )
    data = pd.read_csv(CSV_FILE)
    return data


def evaluate_cv(model, X, y):
    """Leave-One-Out CV is useful here because the dataset is very small."""
    loo = LeaveOneOut()
    predictions = []
    true_values = []

    X = np.asarray(X)
    y = np.asarray(y)

    for train_idx, test_idx in loo.split(X):
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])[0]
        predictions.append(pred)
        true_values.append(y[test_idx][0])

    mse = mean_squared_error(true_values, predictions)
    mae = mean_absolute_error(true_values, predictions)
    return mse, mae


def task1_polynomial_vs_neural_network(data):
    print("=" * 70)
    print("TASK 1: Polynomial regression vs neural network")
    print("=" * 70)

    X = data[["speed_kmh"]].values
    y = data["fuel_consumption_l_per_100km"].values

    results = []

    # Polynomial regression: test degrees from 1 to 5.
    for degree in range(1, 6):
        poly_model = Pipeline([
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("reg", LinearRegression())
        ])
        mse, mae = evaluate_cv(poly_model, X, y)
        results.append({"model": f"Polynomial degree {degree}", "mse": mse, "mae": mae})

    # Neural network model.
    nn_model = Pipeline([
        ("x_scaler", StandardScaler()),
        ("nn", TargetScalerRegressor(
            MLPRegressor(
                hidden_layer_sizes=(16, 8),
                activation="relu",
                solver="lbfgs",
                alpha=0.01,
                max_iter=5000,
                random_state=42
            )
        ))
    ])
    mse, mae = evaluate_cv(nn_model, X, y)
    results.append({"model": "Neural network MLP", "mse": mse, "mae": mae})

    results_df = pd.DataFrame(results).sort_values("mse")
    print("\nModel comparison by Leave-One-Out Cross Validation:")
    print(results_df.to_string(index=False))

    best_poly_row = results_df[results_df["model"].str.contains("Polynomial")].iloc[0]
    best_degree = int(best_poly_row["model"].split()[-1])

    final_poly_model = Pipeline([
        ("poly", PolynomialFeatures(degree=best_degree, include_bias=False)),
        ("reg", LinearRegression())
    ])
    final_poly_model.fit(X, y)

    nn_model.fit(X, y)

    speeds_to_predict = np.array([[35], [95], [140]])
    poly_predictions = final_poly_model.predict(speeds_to_predict)
    nn_predictions = nn_model.predict(speeds_to_predict)

    print(f"\nBest polynomial degree: {best_degree}")
    print("\nPredictions for 35, 95, 140 km/h:")
    for speed, poly_pred, nn_pred in zip(speeds_to_predict.ravel(), poly_predictions, nn_predictions):
        print(
            f"Speed {speed:>3} km/h -> "
            f"Polynomial: {poly_pred:.2f} l/100km, "
            f"Neural network: {nn_pred:.2f} l/100km"
        )

    print("\nConclusion:")
    best_model = results_df.iloc[0]
    print(
        f"The best model by MSE is: {best_model['model']} "
        f"(MSE={best_model['mse']:.4f}, MAE={best_model['mae']:.4f})."
    )
    print(
        "Because the dataset is very small, polynomial regression is usually more stable. "
        "The neural network can work, but it needs more real data for reliable predictions."
    )


def create_extended_dataset(data):
    """
    Creates an extended dataset for Task 2.
    In a real project, trip_time_min and engine_type should be measured directly.
    Here we create them for demonstration because the original CSV contains only speed and fuel consumption.
    """
    extended = data.copy()

    # Assume the tested road section is 100 km long.
    distance_km = 100
    extended["trip_time_min"] = distance_km / extended["speed_kmh"] * 60

    # Add example engine types.
    engine_types = ["petrol", "petrol", "diesel", "diesel", "petrol", "diesel", "petrol", "diesel", "petrol", "diesel"]
    extended["engine_type"] = engine_types

    # Slightly adjust fuel consumption to make engine type meaningful in the example.
    # Diesel is usually more economical, so we subtract a small value for demo data.
    extended["fuel_consumption_l_per_100km"] = np.where(
        extended["engine_type"] == "diesel",
        extended["fuel_consumption_l_per_100km"] - 0.4,
        extended["fuel_consumption_l_per_100km"]
    )

    extended.to_csv("fuel_consumption_extended.csv", index=False)
    return extended


def evaluate_cv_dataframe(model, X_df, y):
    loo = LeaveOneOut()
    predictions = []
    true_values = []

    for train_idx, test_idx in loo.split(X_df):
        X_train = X_df.iloc[train_idx]
        X_test = X_df.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model.fit(X_train, y_train)
        pred = model.predict(X_test)[0]
        predictions.append(pred)
        true_values.append(y_test.iloc[0])

    mse = mean_squared_error(true_values, predictions)
    mae = mean_absolute_error(true_values, predictions)
    return mse, mae


def task2_speed_time_engine(data):
    print("\n" + "=" * 70)
    print("TASK 2: Add trip time and engine type")
    print("=" * 70)

    extended = create_extended_dataset(data)
    print("\nExtended dataset saved as fuel_consumption_extended.csv")
    print(extended.to_string(index=False))

    X = extended[["speed_kmh", "trip_time_min", "engine_type"]]
    y = extended["fuel_consumption_l_per_100km"]

    numeric_features = ["speed_kmh", "trip_time_min"]
    categorical_features = ["engine_type"]

    preprocessor_for_poly = ColumnTransformer([
        ("num_poly", Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler())
        ]), numeric_features),
        ("cat", make_one_hot_encoder(), categorical_features)
    ])

    poly_extended_model = Pipeline([
        ("preprocessor", preprocessor_for_poly),
        ("reg", LinearRegression())
    ])

    preprocessor_for_nn = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", make_one_hot_encoder(), categorical_features)
    ])

    nn_extended_model = Pipeline([
        ("preprocessor", preprocessor_for_nn),
        ("nn", TargetScalerRegressor(
            MLPRegressor(
                hidden_layer_sizes=(16, 8),
                activation="relu",
                solver="lbfgs",
                alpha=0.01,
                max_iter=5000,
                random_state=42
            )
        ))
    ])

    models = {
        "Polynomial regression with speed + time + engine": poly_extended_model,
        "Neural network with speed + time + engine": nn_extended_model,
    }

    print("\nExtended model comparison:")
    comparison = []
    for name, model in models.items():
        mse, mae = evaluate_cv_dataframe(model, X, y)
        comparison.append({"model": name, "mse": mse, "mae": mae})

    comparison_df = pd.DataFrame(comparison).sort_values("mse")
    print(comparison_df.to_string(index=False))

    # Fit final models and predict new examples.
    poly_extended_model.fit(X, y)
    nn_extended_model.fit(X, y)

    new_cars = pd.DataFrame({
        "speed_kmh": [35, 95, 140],
        "trip_time_min": [100 / 35 * 60, 100 / 95 * 60, 100 / 140 * 60],
        "engine_type": ["petrol", "diesel", "petrol"]
    })

    print("\nPredictions for new examples:")
    print(new_cars.to_string(index=False))

    poly_preds = poly_extended_model.predict(new_cars)
    nn_preds = nn_extended_model.predict(new_cars)

    for i, row in new_cars.iterrows():
        print(
            f"Speed {row['speed_kmh']:>3} km/h, "
            f"time {row['trip_time_min']:.1f} min, "
            f"engine {row['engine_type']} -> "
            f"Polynomial: {poly_preds[i]:.2f} l/100km, "
            f"Neural network: {nn_preds[i]:.2f} l/100km"
        )


if __name__ == "__main__":
    original_data = load_original_data()
    print("Original dataset:")
    print(original_data.to_string(index=False))

    task1_polynomial_vs_neural_network(original_data)
    task2_speed_time_engine(original_data)
