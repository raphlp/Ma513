# src/main.py
import csv
import os
import joblib
import time
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, f1_score, average_precision_score, roc_auc_score,
    classification_report, confusion_matrix
)

from models import get_models
from utils import (
    load_creditcard, build_preprocessor, time_based_split,
    fit_eval, ensure_result_dir, RESULT_DIR
)
from rebalance import apply_rebalance


def eval_pipeline(pipe, X_train, y_train, X_test, y_test, tag: str):
    """Evaluate a fitted pipeline (used for rebalanced runs)."""
    y_pred = pipe.predict(X_test)
    y_pred_train = pipe.predict(X_train)

    # Probabilities if available
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        y_proba = pipe.predict_proba(X_test)[:, 1]
    else:
        y_proba = None

    acc = accuracy_score(y_test, y_pred)
    f1_train = f1_score(y_train, y_pred_train)
    f1_test = f1_score(y_test, y_pred)
    ap = average_precision_score(y_test, y_proba) if y_proba is not None else float("nan")
    roc = roc_auc_score(y_test, y_proba) if y_proba is not None else float("nan")
    rep = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "name": tag, "pipeline": pipe, "acc": acc, "f1_train": f1_train, "f1_test": f1_test, "ap": ap, "roc": roc,
        "train_time_s": None, "report": rep, "cm": cm
    }


def run_baseline():
    """Mode a: train models without any rebalance."""
    
    print("\nLoading and pre-processing the data.")
    X, y = load_creditcard()
    preproc = build_preprocessor(X)
    X_train, X_test, y_train, y_test = time_based_split(X, y, test_size=0.2)

    models = get_models(use_class_weight=False)
    results = {}
    best_key, best_f1_train, best_f1_test = [None, None], -1.0, -1.0

    print("Starting to evaluate models.")
    for name, model in models.items():
        out = fit_eval(model, preproc, X_train, y_train, X_test, y_test, name=f"{name}[baseline]")
        print(f"Model {name} evaluated.")
        results[out["name"]] = out
        
        if out["f1_train"] > best_f1_train:
            best_f1_train, best_key[0] = out["f1_train"], out["name"]
        
        if out["f1_test"] > best_f1_test:
            best_f1_test, best_key[1] = out["f1_test"], out["name"]
            
    return results, best_key


def run_rebalanced(strategy: str = "smote", k_neighbors: int = 5):
    """Mode b: train models with a rebalance strategy (SMOTE by default)."""
    
    print(f"\nLoading and pre-processing the data for the strategy {strategy}.")
    X, y = load_creditcard()
    preproc = build_preprocessor(X)
    X_train, X_test, y_train, y_test = time_based_split(X, y, test_size=0.2)

    models = get_models(use_class_weight=False)
    results = {}
    best_key, best_f1_train, best_f1_test = [None, None], -1.0, -1.0

    print("Starting to evaluate models.")
    for name, base_model in models.items():
        pipe = apply_rebalance(base_model, preproc, strategy=strategy, k_neighbors=k_neighbors)
        t0 = time.time()
        pipe.fit(X_train, y_train)
        t1 = time.time()
        print(f"Model {name} evaluated.")

        tag = f"{name}[{strategy}]"
        out = eval_pipeline(pipe, X_train, y_train, X_test, y_test, tag)
        out["train_time_s"] = t1 - t0
        results[tag] = out

        if out["f1_train"] > best_f1_train:
            best_f1_train, best_key[0] = out["f1_train"], tag

        if out["f1_test"] > best_f1_test:
            best_f1_test, best_key[1] = out["f1_test"], tag

    return results, best_key

def save_outputs(results: dict, best_key: str, mode_label: str):
    """Save metrics CSV, best model, and a comparison plot into result/."""
    ensure_result_dir()

    # --- Save metrics CSV ---
    metrics_csv = RESULT_DIR / f"metrics_{mode_label}.csv"
    with open(metrics_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "accuracy", "f1_train", "f1_test", "avg_precision", "roc_auc", "train_time_s"])
        for k, r in results.items():
            w.writerow([
                r["name"],
                f"{r['acc']:.6f}",
                f"{r['f1_train']:.6f}",
                f"{r['f1_test']:.6f}",
                _fmt(r['ap']),
                _fmt(r['roc']),
                _fmt(r['train_time_s'])
            ])
    print(f"Saved metrics → {metrics_csv}\n")

    # --- Save best trained model on testing set ---
    best_pipe = results[best_key[1]]["pipeline"]
    model_path = RESULT_DIR / f"best_model_{_sanitize(best_key[1])}.joblib"
    joblib.dump(best_pipe, model_path)
    print(f"Saved best model → {model_path}\n")
    print(f"Best on training set: {best_key[0]} | F1: {results[best_key[0]]['f1_train']:.4f}")
    print(f"Best on testing set: {best_key[1]} | F1: {results[best_key[1]]['f1_test']:.4f}\n")

    # --- Save a simple F1 bar chart ---
    try:
        plot_path = RESULT_DIR / f"plot_{mode_label}_f1.png"
        df_plot = pd.DataFrame([
                    {"name": r["name"],
                        "f1_train": r["f1_train"],
                        "f1_test": r["f1_test"],
                        "train_time_s": r["train_time_s"]}
                    for r in results.values()
                    ])
        titles = ["Training set", "Testing set"]
        
        _plot_f1_bar(df_plot, plot_path, mode_label, titles)
        print(f"Saved plot → {plot_path}")
    except Exception as e:
        print(f"(Plot skipped: {e})")

def _fmt(x):
    try:
        return f"{x:.6f}"
    except Exception:
        return ""

def _sanitize(s: str) -> str:
    return s.replace(" ", "_").replace("[", "_").replace("]", "_").replace("/", "_")

def simple_menu():
    print("\n=== Fraud Detection Menu ===")
    print("A) Baseline (no rebalance)")
    print("B) Rebalanced (SMOTE oversampling)")
    print("C) Rebalanced (Random undersampling)")
    print("D) Compare A, B and C")
    print("E) Quick demo (show last saved metrics, no training)")
    choice = input("Choose [A/B/C/D/E]: ").strip().lower()
    if choice not in {"a", "b", "c", "d", "e"}:
        print("Invalid choice. Defaulting to D (compare all).")
        choice = "d"

    # ask k only if SMOTE might be used
    k = 5
    if choice in {"b", "d"}:
        k_in = input("SMOTE k_neighbors (default 5): ").strip()
        if k_in.isdigit():
            k = int(k_in)

    print("Do you want to display the confusion matrices ?")
    display = input("[Y/N] ? ").strip().lower()

    return [choice, display], k

def find_latest_metrics():
    files = sorted(glob.glob(str(RESULT_DIR / "metrics_*.csv")))
    return files[-1] if files else None

def print_metrics_csv(path: str):
    df = pd.read_csv(path)
    print(f"\n=== Metrics file: {path} ===")
    cols = [c for c in ["name","f1_train","f1_test","roc_auc","avg_precision","accuracy","train_time_s"] if c in df.columns]
    print(df[cols].to_string(index=False))
    if "f1_test" in df.columns:
        best_row = df.iloc[df["f1_test"].idxmax()]
        print(f"\nBest by F1 on testing set: {best_row['name']} (F1={best_row['f1_test']:.4f})")

    # Save a plot next to the CSV if not already there
    out_png = path.replace("metrics_", "plot_").replace(".csv", "_f1_test.png")
    try:
        _plot_f1_bar(df, out_png, mode_label = "quick demo",titles=["Training set", "Testing set"])
        print(f"Saved plot   → {out_png}\n")
    except Exception as e:
        print(f"(Plot skipped: {e})")

def _results_to_df(results: dict) -> pd.DataFrame:
    """Convert the results dict into a tidy DataFrame (name, f1, roc, ap, acc, time)."""
    rows = []
    for r in results.values():
        rows.append({
            "name": r["name"],
            "f1": r["f1"],
            "roc_auc": r["roc"],
            "avg_precision": r["ap"],
            "accuracy": r["acc"],
            "train_time_s": r["train_time_s"] if r["train_time_s"] is not None else ""
        })
    return pd.DataFrame(rows)

def _plot_f1_bar(df: pd.DataFrame, out_path: str, mode_label: str, titles: list = ["Training set", "Testing set"]):
    """Create a simple bar chart of F1 by model name."""
        
    fig,axes = plt.subplots(2,1,figsize=(9, 9.6))
    
    metrics = ["f1_train", "f1_test"]
    
    for ax, title, metric in zip(axes, titles, metrics):
        # Sorting the data for the corresponding metric
        df_sorted = df.sort_values(metric, ascending=False)
        x = range(len(df_sorted))

        bars = ax.bar(df_sorted["name"], df_sorted[metric], color="steelblue")
        ax.set_title(f"F1-score comparison on {title} with training time — {mode_label}")
        ax.set_ylabel("F1-score")
        ax.set_xticks(x)
        ax.set_xticklabels(df_sorted["name"], rotation=30, ha="right")

        # Training time for the models
        for bar, t in zip(bars, df_sorted["train_time_s"]):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.001, 
                f"{t:.2f}s", 
                ha="center",
                va="bottom",
                fontsize=9,
                color="black"
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def _plot_confusion_matrix(results: dict, mode_label: str):
    """Plot confusion matrices for many models."""
    n_models = len(results)
    
    sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6), constrained_layout = True)
    
    out_path = RESULT_DIR / f"plot_{mode_label}_confusion_matrix.png"
    
    for ax, (model_name, r) in zip(axes, results.items()):
        cm = r["cm"]
        cm_normalized = cm / cm.sum(axis=1, keepdims=True)  # percentage per line

        # Heatmap 
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            cbar=False,
            square = True,
            linewidths=1.5,
            linecolor="white",
            xticklabels=["Non fraud", "Fraud"],
            yticklabels=["Non fraud", "Fraud"],
            ax=ax
        )
        ax.set_title(f"{model_name}\nConfusion matrix — {mode_label}", fontsize=13, fontweight="semibold", pad=15)
        
        ax.set_xlabel("Prediction", fontsize=11)
        ax.set_ylabel("True class", fontsize=11)

    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot   → {out_path}\n")

# --- replace your current main() with this ---
def main():
    [choice, display], k = simple_menu()

    results = None 
    mode_label = None 

    if choice == "a":  # Baseline
        results, best_key = run_baseline()
        mode_label = "a_baseline"
        save_outputs(results, best_key, mode_label)

    elif choice == "b":  # SMOTE
        results, best_key = run_rebalanced(strategy="smote", k_neighbors=k)
        mode_label=f"b_smote_k{k}"
        save_outputs(results, best_key, mode_label)

    elif choice == "c":  # Undersampling
        results, best_key = run_rebalanced(strategy="under", k_neighbors=k)
        mode_label="c_under"
        save_outputs(results, best_key, mode_label)

    elif choice == "d":  # Compare the three (A, B, C)
        res_a, _ = run_baseline()
        res_b, _ = run_rebalanced(strategy="smote", k_neighbors=k)
        res_c, _ = run_rebalanced(strategy="under", k_neighbors=k)
        # merge results
        results = {}
        results.update(res_a)
        results.update(res_b)
        results.update(res_c)
        best_key = [max(results.keys(), key=lambda x: results[x]["f1_train"]),
                    max(results.keys(), key=lambda x: results[x]["f1_test"])]
        mode_label=f"d_compare_all_k{k}"
        save_outputs(results, best_key, mode_label)
    
    elif choice == "e":  # Quick demo (no training)
        latest = find_latest_metrics()
        if latest:
            print_metrics_csv(latest)
        else:
            print("No metrics found in result/. Run A/B/C/D first to generate metrics.")
            
    if display == "y" and results is not None and mode_label is not None:
        _plot_confusion_matrix(results=results, mode_label=mode_label)

if __name__ == "__main__":
    main()