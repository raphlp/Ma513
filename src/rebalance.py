# src/rebalance.py
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def apply_rebalance(model, preproc, strategy: str = "smote", k_neighbors: int = 5):
    """
    Build a pipeline with preprocessing + chosen rebalance step + model.
    strategy: "smote" or "under"
    """
    s = strategy.lower()
    if s == "smote":
        return ImbPipeline([
            ("preproc", preproc),
            ("smote", SMOTE(random_state=42, k_neighbors=k_neighbors)),
            ("clf", model),
        ])
    elif s == "under":
        return ImbPipeline([
            ("preproc", preproc),
            ("under", RandomUnderSampler(random_state=42)),
            ("clf", model),
        ])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")