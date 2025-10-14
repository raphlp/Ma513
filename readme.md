# Fraud Detection Project


This project explores and compares several ensemble learning methods — Random Forest, Gradient Boosting, and XGBoost — for the detection of financial fraud.
The chosen dataset is the Credit Card Fraud Detection Dataset (Kaggle) (`creditcard.csv`), which contains over 280,000 real transactions, among which only 0.17% are fraudulent.
This extreme imbalance makes it a perfect case to study the impact of resampling techniques on model performance.

## Objectives
    •    Evaluate the performance of ensemble models on highly imbalanced financial data.
    •    Experiment with different rebalancing strategies, including SMOTE oversampling and random undersampling.
    •    Compare models using relevant metrics beyond accuracy: F1-score, ROC-AUC, and Precision-Recall AUC.
    •    Analyze trade-offs between false positives and false negatives, emphasizing interpretability for fraud detection use cases.

## Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/raphlp/Ma513.git
cd Ma513
pip install -r requirements.txt
```

## Project Structure

```text
├── data/
│   └── creditcard.csv              # Kaggle Credit Card Fraud Detection dataset
│
├── src/
│   ├── models.py                   # Ensemble model definitions (RF, HGB, XGBoost)
│   ├── utils.py                    # Data loading, preprocessing, metrics
│   ├── rebalance.py                # SMOTE and undersampling pipeline builders
│   └── main.py                     # Main script with interactive menu (A–E modes)
│
├── result/
│   ├── metrics_*.csv               # Saved metrics per experiment
│   ├── plot_*.png                  # Automatic F1-score comparison charts
│   └── best_model_*.joblib         # Serialized trained models
│
├── requirements.txt                # Dependencies list
└── README.md
```

## Usage

Launch the project interactively:
python src/main.py

You will be prompted to choose among the following options:
    •    A) Baseline — train ensemble models without rebalancing
    •    B) SMOTE — train with synthetic oversampling
    •    C) Undersampling — reduce majority class
    •    D) Compare — run and compare A, B, and C automatically
    •    E) Quick demo — display the latest saved results without retraining

All outputs (metrics, plots, and models) are saved in the result/ directory.

## Evaluation

    •    The best configuration obtained was Random Forest + SMOTE, with:
    •    F1-score: 0.8397
    •    ROC-AUC: 0.9786
    •    Precision-Recall AUC: 0.819
    •    Undersampling produced faster training but significantly lower F1-scores.
    •    XGBoost remained stable across all scenarios but required longer training.

## Conclusion

This study confirms that ensemble learning techniques are highly effective for fraud detection tasks when combined with appropriate data rebalancing.
While Random Forest achieved the best compromise between precision and recall, SMOTE proved to be the most beneficial strategy for mitigating extreme class imbalance. Read the report to see more details.
