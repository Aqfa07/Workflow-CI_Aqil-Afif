"""
modelling_tuning.py
===================
Model Training & Hyperparameter Tuning Pipeline - Telco Customer Churn
Author  : Aqil Afif
Project : Submission Akhir MLOps - Dicoding

Kriteria 2 - Advance:
  - Algoritma: Random Forest & XGBoost (dibandingkan)
  - Tuning: GridSearchCV
  - Tracking: MLflow manual logging -> DagsHub
  - Artefak: confusion matrix, ROC curve, feature importance, metric_info.json,
             classification_report.txt
"""

import os
import sys
import json
import shutil
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (aman untuk CI/CD)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve,
)
import mlflow
import mlflow.sklearn

# Dihapus: Import automate pipeline karena membaca data langsung dari file CSV.

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI DAGSHUB & MLFLOW
# Ganti placeholder berikut dengan nilai Anda:
#   DAGSHUB_USER  = username DagsHub Anda
#   DAGSHUB_REPO  = nama repositori DagsHub Anda
#   DAGSHUB_TOKEN = token akses DagsHub (Settings > Tokens)
# ─────────────────────────────────────────────────────────────────────────────
DAGSHUB_USER  = os.getenv("DAGSHUB_USER",  "Aqfa07")
DAGSHUB_REPO  = os.getenv("DAGSHUB_REPO",  "Eksperimen_SML_Aqil-Afif")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN", "ea42d42fe510ecacec023e90a6031c6fa9a869b7")

MLFLOW_TRACKING_URI = (
    f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow"
)
EXPERIMENT_NAME = "Telco_Churn_Classification"

# Direktori artefak lokal sementara
ARTIFACT_DIR = os.path.join("artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SETUP LOGGER
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("logs", "modelling.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger("ModellingPipeline")


# ─────────────────────────────────────────────────────────────────────────────
# SETUP MLFLOW -> DAGSHUB
# ─────────────────────────────────────────────────────────────────────────────
def setup_mlflow() -> None:
    """
    Konfigurasi MLflow untuk tracking ke DagsHub.
    Menggunakan dagshub.init() agar artifact store (S3-compatible)
    terkonfigurasi dengan benar — tanpa ini, mlflow.sklearn.log_model()
    akan gagal upload model binary secara silent.
    """
    import dagshub
    
    # Set token to prevent OAuth browser prompt in CI/CD
    if DAGSHUB_TOKEN:
        os.environ["DAGSHUB_USER_TOKEN"] = DAGSHUB_TOKEN
        dagshub.auth.add_app_token(DAGSHUB_TOKEN)
        
    # dagshub.init() mengkonfigurasi MLFLOW_TRACKING_URI + S3 artifact store credentials
    dagshub.init(
        repo_owner=DAGSHUB_USER,
        repo_name=DAGSHUB_REPO,
        mlflow=True,
    )
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info(f"MLflow Tracking URI : {MLFLOW_TRACKING_URI}")
    logger.info(f"MLflow Experiment   : {EXPERIMENT_NAME}")


# ─────────────────────────────────────────────────────────────────────────────
# FUNGSI PEMBUAT ARTEFAK
# ─────────────────────────────────────────────────────────────────────────────
def save_confusion_matrix(y_true, y_pred, model_name: str) -> str:
    """Simpan plot confusion matrix ke file PNG, return path-nya."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    path = os.path.join(ARTIFACT_DIR, f"confusion_matrix_{model_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Artefak disimpan: {path}")
    return path


def save_roc_curve(y_true, y_prob, model_name: str, auc: float) -> str:
    """Simpan plot ROC Curve ke file PNG, return path-nya."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#4C72B0", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - {model_name}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(ARTIFACT_DIR, f"roc_curve_{model_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Artefak disimpan: {path}")
    return path


def save_feature_importance(model, feature_names: list, model_name: str) -> str:
    """Simpan plot feature importance (top-20), return path-nya."""
    importances = model.feature_importances_
    feat_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(20)
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.barplot(data=feat_df, x="importance", y="feature", palette="viridis", ax=ax)
    ax.set_title(f"Top-20 Feature Importance - {model_name}")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    path = os.path.join(ARTIFACT_DIR, f"feature_importance_{model_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Artefak disimpan: {path}")
    return path


def save_metric_json(metrics: dict, params: dict, model_name: str) -> str:
    """Simpan ringkasan metrik + parameter ke metric_info.json, return path-nya."""
    payload = {"model": model_name, "best_params": params, "metrics": metrics}
    path = os.path.join(ARTIFACT_DIR, f"metric_info_{model_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
    logger.info(f"Artefak disimpan: {path}")
    return path


def save_classification_report(y_true, y_pred, model_name: str) -> str:
    """Simpan classification report sebagai .txt, return path-nya."""
    report = classification_report(y_true, y_pred, target_names=["No Churn", "Churn"])
    path = os.path.join(ARTIFACT_DIR, f"classification_report_{model_name}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Classification Report - {model_name}\n")
        f.write("=" * 50 + "\n")
        f.write(report)
    logger.info(f"Artefak disimpan: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# FUNGSI UTAMA: TRAINING + TUNING + LOGGING
# ─────────────────────────────────────────────────────────────────────────────
def run_experiment(
    model,
    param_grid: dict,
    X_train, X_test,
    y_train, y_test,
    model_name: str,
    feature_names: list,
) -> dict:
    """
    Menjalankan GridSearchCV, lalu mencatat semua hasil ke MLflow secara manual.

    Returns:
        dict: ringkasan metrik terbaik
    """
    logger.info("=" * 60)
    logger.info(f"EKSPERIMEN: {model_name}")
    logger.info("=" * 60)

    # ── GridSearchCV ──────────────────────────────────────────────────────────
    logger.info("Menjalankan GridSearchCV ...")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="f1",           # optimasi F1 (lebih cocok untuk data imbalanced)
        cv=5,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )
    grid_search.fit(X_train, y_train)
    best_model  = grid_search.best_estimator_
    best_params = grid_search.best_params_
    logger.info(f"Best params: {best_params}")
    logger.info(f"Best CV F1 : {grid_search.best_score_:.4f}")

    # ── Evaluasi pada data uji ────────────────────────────────────────────────
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy"  : float(accuracy_score(y_test, y_pred)),
        "f1_score"  : float(f1_score(y_test, y_pred)),
        "precision" : float(precision_score(y_test, y_pred)),
        "recall"    : float(recall_score(y_test, y_pred)),
        "roc_auc"   : float(roc_auc_score(y_test, y_prob)),
        "cv_best_f1": float(grid_search.best_score_),
    }
    logger.info(f"Metrik: {metrics}")

    # ── Buat artefak lokal ────────────────────────────────────────────────────
    cm_path      = save_confusion_matrix(y_test, y_pred, model_name)
    roc_path     = save_roc_curve(y_test, y_prob, model_name, metrics["roc_auc"])
    fi_path      = save_feature_importance(best_model, feature_names, model_name)
    json_path    = save_metric_json(metrics, best_params, model_name)
    report_path  = save_classification_report(y_test, y_pred, model_name)

    # ── MLflow: Manual Logging ────────────────────────────────────────────────
    with mlflow.start_run(run_name=model_name):

        # 1. Log tag
        mlflow.set_tag("model_name",  model_name)
        mlflow.set_tag("author",      "Aqil Afif")
        mlflow.set_tag("dataset",     "Telco Customer Churn")
        mlflow.set_tag("tuning",      "GridSearchCV")

        # 2. Log hyperparameter terbaik hasil tuning
        for param, value in best_params.items():
            mlflow.log_param(param, value)
        mlflow.log_param("cv_folds",    5)
        mlflow.log_param("test_size",   0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("n_features",  len(feature_names))

        # 3. Log metrik
        mlflow.log_metric("accuracy",   metrics["accuracy"])
        mlflow.log_metric("f1_score",   metrics["f1_score"])
        mlflow.log_metric("precision",  metrics["precision"])
        mlflow.log_metric("recall",     metrics["recall"])
        mlflow.log_metric("roc_auc",    metrics["roc_auc"])
        mlflow.log_metric("cv_best_f1", metrics["cv_best_f1"])

        # 4. Log model (sklearn flavor)
        # Catatan: input_example dihapus karena dapat menyebabkan silent failure
        # saat schema inference gagal di DagsHub artifact store.
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
        )

        # 4b. Simpan model lokal sebagai fallback (untuk Docker build di CI/CD)
        local_save_path = os.path.join("model_local", model_name)
        if os.path.exists(local_save_path):
            shutil.rmtree(local_save_path)
        mlflow.sklearn.save_model(sk_model=best_model, path=local_save_path)
        logger.info(f"Model disimpan lokal: {local_save_path}")

        # 5. Log artefak tambahan (minimal 2, kita log 5)
        mlflow.log_artifact(cm_path,     artifact_path="plots")
        mlflow.log_artifact(roc_path,    artifact_path="plots")
        mlflow.log_artifact(fi_path,     artifact_path="plots")
        mlflow.log_artifact(json_path,   artifact_path="reports")
        mlflow.log_artifact(report_path, artifact_path="reports")

        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow Run ID: {run_id}")

    logger.info(f"Eksperimen '{model_name}' selesai.\n")
    return {"model_name": model_name, "run_id": run_id, **metrics}


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── 1. Load data hasil preprocessing ──────────────────────────────────────
    logger.info("Memuat dataset hasil preprocessing ...")
    train_df = pd.read_csv(os.path.join("telco_preprocessing", "train.csv"))
    test_df  = pd.read_csv(os.path.join("telco_preprocessing", "test.csv"))
    
    X_train = train_df.drop("Churn", axis=1)
    y_train = train_df["Churn"]
    X_test  = test_df.drop("Churn", axis=1)
    y_test  = test_df["Churn"]
    
    feature_names = list(X_train.columns)
    logger.info(f"Data siap: train={X_train.shape}, test={X_test.shape}")

    # ── 2. Setup MLflow -> DagsHub ────────────────────────────────────────────
    setup_mlflow()

    results = []

    # ── 3. Eksperimen A: Random Forest + GridSearchCV ─────────────────────────
    rf_param_grid = {
        "n_estimators"     : [100, 200],
        "max_depth"        : [None, 10, 20],
        "min_samples_split": [2, 5],
        "class_weight"     : ["balanced"],
    }
    rf_result = run_experiment(
        model=RandomForestClassifier(random_state=42),
        param_grid=rf_param_grid,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        model_name="RandomForest",
        feature_names=feature_names,
    )
    results.append(rf_result)

    # ── 4. Eksperimen B: XGBoost + GridSearchCV ───────────────────────────────
    try:
        from xgboost import XGBClassifier  # opsional

        xgb_param_grid = {
            "n_estimators"  : [100, 200],
            "max_depth"     : [3, 6],
            "learning_rate" : [0.05, 0.1],
            "scale_pos_weight": [
                int((y_train == 0).sum() / (y_train == 1).sum())
            ],
        }
        xgb_result = run_experiment(
            model=XGBClassifier(
                random_state=42, eval_metric="logloss", use_label_encoder=False
            ),
            param_grid=xgb_param_grid,
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
            model_name="XGBoost",
            feature_names=feature_names,
        )
        results.append(xgb_result)
    except ImportError:
        logger.warning("XGBoost tidak tersedia, lewati eksperimen XGBoost.")

    # ── 5. Rangkuman perbandingan ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("RANGKUMAN PERBANDINGAN MODEL")
    print("=" * 65)
    results_df = pd.DataFrame(results)
    summary_df = results_df.set_index("model_name").drop(columns=["run_id"])
    print(summary_df.round(4).to_string())
    print("=" * 65)

    best_model_name = summary_df["f1_score"].idxmax()
    print(f"\nModel terbaik (F1): {best_model_name}")
    print(f"F1 Score           : {summary_df.loc[best_model_name, 'f1_score']:.4f}")
    print(f"ROC-AUC            : {summary_df.loc[best_model_name, 'roc_auc']:.4f}")
    print(f"\nCek hasil di DagsHub MLflow:\n{MLFLOW_TRACKING_URI}")

    # ── 6. Salin model terbaik ke ./model/ (digunakan Docker build di CI/CD) ──
    best_result = results_df.loc[results_df["f1_score"].idxmax()]
    best_run_id  = best_result["run_id"]
    best_local   = os.path.join("model_local", best_model_name)

    final_model_path = "model"
    if os.path.exists(final_model_path):
        shutil.rmtree(final_model_path)
    shutil.copytree(best_local, final_model_path)
    logger.info(f"Model terbaik disalin ke '{final_model_path}/' dari '{best_local}/'")
    print(f"\nModel terbaik disimpan di: ./{final_model_path}/")

    # ── 7. Simpan best run_id ke file (digunakan CI/CD) ───────────────────────
    run_id_path = "run_id.txt"
    with open(run_id_path, "w", encoding="utf-8") as f:
        f.write(best_run_id)
    logger.info(f"Best run ID disimpan ke '{run_id_path}': {best_run_id}")
    print(f"Best Run ID : {best_run_id}")
    print(f"Tersimpan di: {run_id_path}")
