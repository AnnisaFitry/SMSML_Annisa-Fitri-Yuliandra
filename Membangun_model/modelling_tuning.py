import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.svm import SVC
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def train_tuning():

    # ==================================================
    # MLflow Tracking Server
    # ==================================================

    mlflow.set_tracking_uri(
        "http://127.0.0.1:5000"
    )

    mlflow.set_experiment(
        "Telco Churn Tuning"
    )

    # ==================================================
    # Load Dataset
    # ==================================================

    df = pd.read_csv(
        "churn_clean.csv"
    )

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # ==================================================
    # Train Test Split
    # ==================================================

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
    )

    # ==================================================
    # Hyperparameter Tuning
    # ==================================================

    param_grid = {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"]
    }

    grid_search = GridSearchCV(
        estimator=SVC(),
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid_search.fit(
        X_train,
        y_train
    )

    best_model = grid_search.best_estimator_

    # ==================================================
    # Training Prediction
    # ==================================================

    train_pred = best_model.predict(
        X_train
    )

    # ==================================================
    # Testing Prediction
    # ==================================================

    test_pred = best_model.predict(
        X_test
    )

    # ==================================================
    # Training Metrics
    # ==================================================

    train_accuracy = accuracy_score(
        y_train,
        train_pred
    )

    train_precision = precision_score(
        y_train,
        train_pred
    )

    train_recall = recall_score(
        y_train,
        train_pred
    )

    train_f1 = f1_score(
        y_train,
        train_pred
    )

    # ==================================================
    # Testing Metrics
    # ==================================================

    test_accuracy = accuracy_score(
        y_test,
        test_pred
    )

    test_precision = precision_score(
        y_test,
        test_pred
    )

    test_recall = recall_score(
        y_test,
        test_pred
    )

    test_f1 = f1_score(
        y_test,
        test_pred
    )

    # ==================================================
    # Manual Logging
    # ==================================================

    with mlflow.start_run():
        mlflow.log_param("tuning", True)
        # ==============================================
        # Dataset Information
        # ==============================================

        mlflow.log_param(
            "dataset_rows",
            len(df)
        )

        mlflow.log_param(
            "dataset_columns",
            df.shape[1]
        )

        mlflow.log_param(
            "train_size",
            len(X_train)
        )

        mlflow.log_param(
            "test_size",
            len(X_test)
        )

        # ==============================================
        # Estimator Information
        # ==============================================

        mlflow.log_param(
            "estimator_name",
            "SVC"
        )

        # ==============================================
        # Best Parameters
        # ==============================================

        for param_name, value in (
            grid_search.best_params_.items()
        ):
            mlflow.log_param(
                param_name,
                value
            )

        # ==============================================
        # Cross Validation Metric
        # ==============================================

        mlflow.log_metric(
            "best_cv_score",
            grid_search.best_score_
        )

        # ==============================================
        # Training Metrics
        # ==============================================

        mlflow.log_metric(
            "training_accuracy_score",
            train_accuracy
        )

        mlflow.log_metric(
            "training_precision_score",
            train_precision
        )

        mlflow.log_metric(
            "training_recall_score",
            train_recall
        )

        mlflow.log_metric(
            "training_f1_score",
            train_f1
        )

        # ==============================================
        # Testing Metrics
        # ==============================================

        mlflow.log_metric(
            "testing_accuracy_score",
            test_accuracy
        )

        mlflow.log_metric(
            "testing_precision_score",
            test_precision
        )

        mlflow.log_metric(
            "testing_recall_score",
            test_recall
        )

        mlflow.log_metric(
            "testing_f1_score",
            test_f1
        )

        # ==============================================
        # Save Best Model
        # ==============================================

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model"
        )

        print("\n===== BEST PARAMETER =====")
        print(grid_search.best_params_)

        print("\n===== TRAIN METRICS =====")
        print("Accuracy :", train_accuracy)
        print("Precision:", train_precision)
        print("Recall   :", train_recall)
        print("F1 Score :", train_f1)

        print("\n===== TEST METRICS =====")
        print("Accuracy :", test_accuracy)
        print("Precision:", test_precision)
        print("Recall   :", test_recall)
        print("F1 Score :", test_f1)


if __name__ == "__main__":
    train_tuning()