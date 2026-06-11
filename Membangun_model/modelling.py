import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def train_model(file_path):

    # ==========================
    # MLflow Tracking
    # ==========================
    mlflow.set_tracking_uri(
        "http://127.0.0.1:5000"
    )

    mlflow.set_experiment(
        "Telco Customer Churn"
    )

    # ==========================
    # Load Data
    # ==========================
    df = pd.read_csv(file_path)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # ==========================
    # Split Data
    # ==========================
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
    )

    # ==========================
    # Training
    # ==========================
    with mlflow.start_run():

        model = SVC(
            kernel="rbf",
            random_state=42
        )
        # ==========================
        # Autolog
        # ==========================
        mlflow.sklearn.autolog()
        
        model.fit(
            X_train,
            y_train
        )

        # ==========================
        # Prediksi
        # ==========================
        y_pred = model.predict(
            X_test
        )

        # ==========================
        # Evaluasi
        # ==========================
        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred
        )

        recall = recall_score(
            y_test,
            y_pred
        )

        f1 = f1_score(
            y_test,
            y_pred
        )

        # ==========================
        # Log Metrics Manual
        # ==========================
        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1_score",
            f1
        )

        print("Accuracy :", accuracy)
        print("Precision:", precision)
        print("Recall   :", recall)
        print("F1 Score :", f1)


if __name__ == "__main__":
    file_path="churn_clean.csv"
    train_model(file_path)