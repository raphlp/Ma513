# first_sim_no_rebal.py
import sys, time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib

def load_npz(path):
    data = np.load(path, allow_pickle=True)
    arr = data['arr']
    X = np.array([x[0].astype('float32').ravel() for x in arr])
    y = np.array([int(x[1]) for x in arr])
    return X, y

def train_and_eval(X_train, y_train, X_test, y_test, clf, name):
    t0 = time.time()
    clf.fit(X_train, y_train)
    t1 = time.time()
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n=== {name} ===")
    print(f"train time: {t1-t0:.1f}s  |  accuracy: {acc:.4f}  |  macro-F1: {f1_macro:.4f}")
    print("classification report:\n", report)
    print("confusion matrix shape:", cm.shape)
    return {'clf': clf, 'acc': acc, 'f1_macro': f1_macro, 'time': t1-t0, 'report': report, 'cm': cm}

def main(npz_path):
    print("Loading", npz_path)
    X, y = load_npz(npz_path)
    print("X shape:", X.shape, "y shape:", y.shape)
    # stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print("Train/test:", X_train.shape, X_test.shape)

    results = {}

    # RandomForest (rapide version)
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
    results['RF'] = train_and_eval(X_train, y_train, X_test, y_test, rf, "RandomForest")

    # HistGradientBoosting (scikit-learn)
    hgb = HistGradientBoostingClassifier(max_iter=200)
    results['HGB'] = train_and_eval(X_train, y_train, X_test, y_test, hgb, "HistGradientBoosting")

    # XGBoost
    xgb = XGBClassifier(n_estimators=150, use_label_encoder=False, eval_metric='mlogloss', verbosity=0)
    results['XGB'] = train_and_eval(X_train, y_train, X_test, y_test, xgb, "XGBoost")

    # save best by macro-F1
    best_name = max(results.keys(), key=lambda k: results[k]['f1_macro'])
    print("\nBest model by macro-F1:", best_name, results[best_name]['f1_macro'])
    joblib.dump(results[best_name]['clf'], f"best_model_{best_name}.joblib")
    print("Saved model:", f"best_model_{best_name}.joblib")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python first_sim_no_rebal.py /path/to/images_malware.npz")
        sys.exit(1)
    main(sys.argv[1])
