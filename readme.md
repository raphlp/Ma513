# Fraud Detection Project

This project implements and compares several **ensemble learning methods** — namely **Random Forest**, **Gradient Boosting**, and **XGBoost** — for **fraud detection** and **imbalanced classification**.  
The dataset used (`images_malware.npz`) contains 25 malware classes of varying frequencies, providing a realistic case of class imbalance.

## Objectives
- Evaluate the performance of ensemble models on a naturally imbalanced dataset.  
- Experiment with different **rebalancing strategies** such as class weighting and oversampling (SMOTE).  
- Analyze performance metrics beyond accuracy — e.g., **macro-F1**, **precision**, **recall**, and **confusion matrices** — to understand the trade-off between false positives and false negatives.  

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
│   └── images_malware.npz         # Original dataset (not modified)
├── src/
│   └── first_sim_no_rebal.py      # Baseline experiment without rebalancing
├── reports/                       # Figures, metrics, and evaluation results
├── requirements.txt               # Dependencies
└── README.md
```
