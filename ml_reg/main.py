import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# File with English levels stored as text values:
# Elementary, Pre-Intermediate, Intermediate, Upper-Intermediate, Advanced
DATA_FILE = "internship_candidates_cefr_final.csv"

# Ordered mapping for English levels.
# Logistic Regression needs numbers, so we convert text levels to ordinal values.
ENGLISH_LEVEL_MAP = {
    "Elementary": 1,
    "Pre-Intermediate": 2,
    "Intermediate": 3,
    "Upper-Intermediate": 4,
    "Advanced": 5,
}


def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV data and prepare EnglishLevel as a numeric feature."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)

    required_columns = [
        "Experience",
        "Grade",
        "EnglishLevel",
        "Age",
        "EntryTestScore",
        "Accepted",
    ]

    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    df["EnglishLevelNumeric"] = df["EnglishLevel"].map(ENGLISH_LEVEL_MAP)

    if df["EnglishLevelNumeric"].isna().any():
        unknown_levels = df.loc[df["EnglishLevelNumeric"].isna(), "EnglishLevel"].unique()
        raise ValueError(f"Unknown English levels: {unknown_levels}")

    return df


def train_model(df: pd.DataFrame):
    """Train Logistic Regression model and print quality metrics."""
    feature_columns = [
        "Experience",
        "Grade",
        "EnglishLevelNumeric",
        "Age",
        "EntryTestScore",
    ]

    X = df[feature_columns]
    y = df["Accepted"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    return model, scaler, feature_columns


def predict_one_student(model, scaler, feature_columns):
    """Example prediction for one student."""
    student = pd.DataFrame(
        [{
            "Experience": 2,
            "Grade": 10.5,
            "EnglishLevelNumeric": ENGLISH_LEVEL_MAP["Upper-Intermediate"],
            "Age": 19,
            "EntryTestScore": 850,
        }]
    )

    student = student[feature_columns]
    student_scaled = scaler.transform(student)

    probability = model.predict_proba(student_scaled)[0][1]
    prediction = model.predict(student_scaled)[0]

    print("\nExample student prediction:")
    print("Probability of acceptance:", round(probability, 3))
    print("Accepted:" if prediction == 1 else "Not accepted")


def plot_acceptance_probability(df: pd.DataFrame, model, scaler, feature_columns):
    """Build a plot of acceptance probability by English level and entry test score."""
    english_levels = list(ENGLISH_LEVEL_MAP.keys())
    english_numeric_values = list(ENGLISH_LEVEL_MAP.values())
    test_scores = np.linspace(df["EntryTestScore"].min(), df["EntryTestScore"].max(), 100)

    # Other features are fixed at their average values.
    average_experience = df["Experience"].mean()
    average_grade = df["Grade"].mean()
    average_age = df["Age"].mean()

    plt.figure(figsize=(10, 6))

    for level_name, level_number in zip(english_levels, english_numeric_values):
        rows = []
        for score in test_scores:
            rows.append({
                "Experience": average_experience,
                "Grade": average_grade,
                "EnglishLevelNumeric": level_number,
                "Age": average_age,
                "EntryTestScore": score,
            })

        plot_data = pd.DataFrame(rows)[feature_columns]
        plot_data_scaled = scaler.transform(plot_data)
        probabilities = model.predict_proba(plot_data_scaled)[:, 1]

        plt.plot(test_scores, probabilities, label=level_name)

    plt.title("Acceptance Probability by English Level and Entry Test Score")
    plt.xlabel("Entry Test Score")
    plt.ylabel("Probability of Acceptance")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend(title="English Level")
    plt.tight_layout()

    output_file = "acceptance_probability.png"
    plt.savefig(output_file, dpi=200)
    plt.show()

    print(f"\nPlot saved as: {output_file}")


def main():
    df = load_data(DATA_FILE)
    model, scaler, feature_columns = train_model(df)
    predict_one_student(model, scaler, feature_columns)
    plot_acceptance_probability(df, model, scaler, feature_columns)


if __name__ == "__main__":
    main()
