"""
modelling.py
============
Basic Model Training - Telco Customer Churn
"""
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

if __name__ == "__main__":
    print("Memuat dataset...")
    train_df = pd.read_csv(os.path.join("telco_preprocessing", "train.csv"))
    test_df  = pd.read_csv(os.path.join("telco_preprocessing", "test.csv"))
    
    X_train = train_df.drop("Churn", axis=1)
    y_train = train_df["Churn"]
    X_test  = test_df.drop("Churn", axis=1)
    y_test  = test_df["Churn"]

    mlflow.autolog()

    with mlflow.start_run(run_name="Basic_RandomForest"):
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Akurasi model dasar: {acc:.4f}")

    print("Model dasar berhasil dilatih dan dicatat oleh MLflow autolog.")
