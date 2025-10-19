from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

def get_models(use_class_weight: bool = False):
    """
    Return base ensemble models.
    If use_class_weight=True, apply class_weight='balanced' where supported.
    For others, you can pass sample_weight at fit time.
    """
    rf_params = dict(n_estimators=300, max_depth=None, n_jobs=-1, random_state=42)
    if use_class_weight:
        rf_params["class_weight"] = "balanced"

    models = {
        "RandomForest": RandomForestClassifier(**rf_params),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.1, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=600, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42, n_jobs=-1, tree_method="hist"
        ),
        "SVM" : SVC(C = 1, kernel = 'rbf', gamma = 'scale', max_iter = 5000)
    }
    return models