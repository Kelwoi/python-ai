import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


# -----------------------------
# Task 1
# -----------------------------
def task_1():
    print("TASK 1: Linear Regression without categorical columns")

    data = pd.read_csv("energy_usage.csv")

    print("\nFirst 5 rows:")
    print(data.head())

    X = data[["temperature", "humidity", "hour", "is_weekend"]]
    y = data["consumption"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    error_percent = mean_absolute_percentage_error(y_test, predictions) * 100

    print(f"\nTask 1 error: {error_percent:.2f}%")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=predictions)
    plt.xlabel("Real consumption")
    plt.ylabel("Predicted consumption")
    plt.title("Task 1: Real vs Predicted Consumption")
    plt.grid(True)
    plt.show()


# -----------------------------
# Task 2
# -----------------------------
def task_2():
    print("\nTASK 2: Linear Regression with categorical columns")

    data = pd.read_csv("energy_usage_plus.csv")

    print("\nFirst 5 rows:")
    print(data.head())

    X = data[[
        "temperature",
        "humidity",
        "season",
        "hour",
        "district_type",
        "is_weekend"
    ]]

    y = data["consumption"]

    numeric_features = [
        "temperature",
        "humidity",
        "hour",
        "is_weekend"
    ]

    categorical_features = [
        "season",
        "district_type"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features)
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regression", LinearRegression())
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    error_percent = mean_absolute_percentage_error(y_test, predictions) * 100

    print(f"\nTask 2 error: {error_percent:.2f}%")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=predictions)
    plt.xlabel("Real consumption")
    plt.ylabel("Predicted consumption")
    plt.title("Task 2: Real vs Predicted Consumption")
    plt.grid(True)
    plt.show()


task_1()
task_2()