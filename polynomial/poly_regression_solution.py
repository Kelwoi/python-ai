import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import LeaveOneOut, cross_val_predict

# 1. Load data
DATA_PATH = "fuel_consumption_vs_speed.csv"
df = pd.read_csv(DATA_PATH)

X = df[["speed_kmh"]].values
y = df["fuel_consumption_l_per_100km"].values

# 2. Compare polynomial regression models
# We use Leave-One-Out Cross Validation because the dataset is very small.
max_degree = 7
results = []

for degree in range(1, max_degree + 1):
    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression()
    )

    # Metrics on the same data after fitting
    model.fit(X, y)
    train_predictions = model.predict(X)
    train_mse = mean_squared_error(y, train_predictions)
    train_mae = mean_absolute_error(y, train_predictions)

    # Cross-validation metrics: better for choosing the real model quality
    cv_predictions = cross_val_predict(model, X, y, cv=LeaveOneOut())
    cv_mse = mean_squared_error(y, cv_predictions)
    cv_mae = mean_absolute_error(y, cv_predictions)

    results.append({
        "degree": degree,
        "train_MSE": train_mse,
        "train_MAE": train_mae,
        "CV_MSE": cv_mse,
        "CV_MAE": cv_mae,
    })

results_df = pd.DataFrame(results)
print("Model comparison:")
print(results_df.to_string(index=False))

# 3. Choose the best degree by the lowest cross-validation MSE
best_degree = int(results_df.loc[results_df["CV_MSE"].idxmin(), "degree"])
print(f"\nBest polynomial degree: {best_degree}")

best_model = make_pipeline(
    PolynomialFeatures(degree=best_degree, include_bias=False),
    LinearRegression()
)
best_model.fit(X, y)

# 4. Predict fuel consumption for required speeds
speeds_to_predict = np.array([[35], [95], [140]])
predictions = best_model.predict(speeds_to_predict)

print("\nPredictions:")
for speed, prediction in zip(speeds_to_predict.flatten(), predictions):
    print(f"Speed {speed} km/h -> {prediction:.2f} l/100 km")

# 5. Create a graph
x_line = np.linspace(X.min(), 140, 300).reshape(-1, 1)
y_line = best_model.predict(x_line)

plt.figure(figsize=(8, 5))
plt.scatter(X, y, label="Real data")
plt.plot(x_line, y_line, label=f"Polynomial regression, degree {best_degree}")
plt.scatter(speeds_to_predict, predictions, marker="x", s=90, label="Predictions")
plt.xlabel("Speed, km/h")
plt.ylabel("Fuel consumption, l/100 km")
plt.title("Fuel consumption vs speed")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("fuel_regression_plot.png", dpi=200)
plt.show()
