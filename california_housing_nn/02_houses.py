import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing


def load_data():
    """Loads and returns the California housing dataset as a Pandas DataFrame."""

    housing = fetch_california_housing()

    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df["Price"] = housing.target

    print("Dataset loaded successfully")
    print("First 5 rows:")
    print(df.head())

    print("\nColumns:")
    print(df.columns)

    print("\nDataset shape:")
    print(df.shape)

    return df


def preprocess_data(df):
    """Splits and preprocesses the data."""

    X = df.drop(columns=["Price"])
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def build_model(input_shape):
    """Builds and returns a compiled neural network model."""

    model = Sequential([
        Dense(128, activation="relu", input_shape=(input_shape,)),
        Dropout(0.2),

        Dense(64, activation="relu"),
        Dropout(0.2),

        Dense(32, activation="relu"),

        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model


def train_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
    """Trains the model and returns the training history."""

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping]
    )

    return history


def evaluate_model(model, X_test, y_test):
    """Evaluates the model and returns the test loss and MAE."""

    loss, mae = model.evaluate(X_test, y_test)

    return loss, mae


def plot_loss(history):
    """Plots the training and validation loss curves."""

    plt.figure(figsize=(12, 5))

    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")

    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.title("Model Training Loss Curve")
    plt.grid(True)

    plt.show()


def plot_predictions(y_test, y_pred):
    """Plots actual vs predicted prices."""

    plt.figure(figsize=(8, 6))

    plt.scatter(y_test, y_pred, alpha=0.5)

    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted House Prices")
    plt.grid(True)

    plt.show()


def print_model_info():
    """Prints information about selected architecture."""

    print("\nSelected neural network architecture:")
    print("Input layer: 8 features")
    print("Hidden layer 1: 128 neurons, ReLU activation")
    print("Dropout: 0.2")
    print("Hidden layer 2: 64 neurons, ReLU activation")
    print("Dropout: 0.2")
    print("Hidden layer 3: 32 neurons, ReLU activation")
    print("Output layer: 1 neuron")
    print("Optimizer: Adam")
    print("Loss function: MSE")
    print("Metric: MAE")


if __name__ == "__main__":
    df = load_data()

    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    model = build_model(X_train.shape[1])

    print_model_info()

    print("\nModel summary:")
    model.summary()

    history = train_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=100,
        batch_size=32
    )

    loss, mae = evaluate_model(model, X_test, y_test)

    print("\nResult:")
    print(f"Test MSE Loss: {loss:.4f}")
    print(f"Test Mean Absolute Error: {mae:.4f}")

    print("\nExplanation:")
    print("The target value in California Housing is measured in hundreds of thousands of dollars.")
    print("For example, MAE = 0.30 means the average error is about 30,000 dollars.")

    plot_loss(history)

    y_pred = model.predict(X_test)
    plot_predictions(y_test, y_pred)