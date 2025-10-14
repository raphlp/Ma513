import os
from pathlib import Path
import time
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, average_precision_score, roc_auc_score,
    precision_recall_curve, classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

# Optional (used in modes c/d)
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    HAS_IMBLEARN = True
except Exception:
    HAS_IMBLEARN = False

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
RESULT_DIR = BASE_DIR / "result"

def load_creditcard(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Load dataset and split features/target."""
    df = pd.read_csv(path)
    y = df["Class"].astype(int)
    X = df.drop(columns=["Class"])
    return X, y

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Scale 'Amount' and 'Time'. Pass other PCA columns as-is."""
    cols = X.columns.tolist()
    scale_cols = [c for c in ["Amount", "Time"] if c in cols]
    transformers = []
    if scale_cols:
        transformers.append(("scale_amt_time", StandardScaler(), scale_cols))
    # passthrough all columns (scikit-learn will drop duplicates automatically)
    preproc = ColumnTransformer(transformers=transformers, remainder="passthrough")
    return preproc

def time_based_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """Chronological split to avoid leakage (uses 'Time'). If 'Time' missing, fallback to stratified split."""
    if "Time" in X.columns:
        order = np.argsort(X["Time"].values)
        Xs, ys = X.iloc[order], y.iloc[order]
        split_idx = int((1.0 - test_size) * len(Xs))
        X_train, X_test = Xs.iloc[:split_idx], Xs.iloc[split_idx:]
        y_train, y_test = ys.iloc[:split_idx], ys.iloc[split_idx:]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
    return X_train, X_test, y_train, y_test

def fit_eval(model, preproc, X_train, y_train, X_test, y_test, sample_weight=None, name="model"):
    """Fit a pipeline and compute metrics. Returns dict with metrics and artifacts."""
    pipe = Pipeline([("preproc", preproc), ("clf", model)])
    t0 = time.time()
    if sample_weight is not None:
        pipe.fit(X_train, y_train, clf__sample_weight=sample_weight)
    else:
        pipe.fit(X_train, y_train)
    t1 = time.time()

    # Predictions
    y_pred = pipe.predict(X_test)
    # Probabilities (if available)
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        y_proba = pipe.predict_proba(X_test)[:, 1]
    elif hasattr(pipe.named_steps["clf"], "decision_function"):
        # Map decision scores to 0-1 (rough). For HGB, predict_proba exists.
        scores = pipe.decision_function(X_test)
        y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    else:
        y_proba = None

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)  # binary F1 (class=1)
    ap = average_precision_score(y_test, y_proba) if y_proba is not None else np.nan
    roc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan

    # Optional extras (string, matrix)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "name": name,
        "pipeline": pipe,
        "acc": acc,
        "f1": f1,
        "ap": ap,
        "roc": roc,
        "train_time_s": t1 - t0,
        "report": report,
        "cm": cm,
    }

def make_smote_pipeline(model, preproc, k_neighbors: int = 5):
    """Preproc -> SMOTE -> clf. Train-time only."""
    if not HAS_IMBLEARN:
        raise RuntimeError("imblearn is not installed. Install 'imbalanced-learn'.")
    return ImbPipeline([
        ("preproc", preproc),
        ("smote", SMOTE(random_state=42, k_neighbors=k_neighbors)),
        ("clf", model),
    ])

def make_under_pipeline(model, preproc):
    """Preproc -> RandomUnderSampler -> clf."""
    if not HAS_IMBLEARN:
        raise RuntimeError("imblearn is not installed. Install 'imbalanced-learn'.")
    return ImbPipeline([
        ("preproc", preproc),
        ("under", RandomUnderSampler(random_state=42)),
        ("clf", model),
    ])

def compute_balanced_sample_weight(y_train: pd.Series):
    """Return per-sample weights for class imbalance."""
    return compute_sample_weight(class_weight="balanced", y=y_train)

def ensure_result_dir():
    os.makedirs(RESULT_DIR, exist_ok=True)