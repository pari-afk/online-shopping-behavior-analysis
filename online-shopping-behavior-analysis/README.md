# Online Shopping Behaviour Analysis

## Overview
This project analyzes online shopping behaviour using both real-world clickstream data and self-reported survey responses. The primary objective is to predict whether an online shopping session will result in a purchase and to explore differences between objective behavioural data and how shoppers perceive their own behaviour.

Two datasets were used:
- The UCI Online Shoppers Purchasing Intention dataset (12,000+ sessions)
- A small, self-collected survey dataset (n = 20)

By comparing these datasets, the project examines potential mismatches between shopping intention and actual behaviour.

---

## Methods
The following machine learning techniques were applied:

### Supervised Learning
- Logistic Regression
- Decision Tree Classification

### Unsupervised Learning
- K-Means Clustering

### Dimensionality Reduction
- Principal Component Analysis (PCA)

A standard train/test split was used for model evaluation. Feature scaling was applied where required (logistic regression and k-means), while tree-based models were trained on unscaled data.

---

## Key Results
- Logistic regression achieved approximately **88% accuracy** on the UCI dataset.
- Decision tree classification achieved approximately **85% accuracy**, with different error trade-offs compared to logistic regression.
- K-means clustering combined with PCA revealed distinct behavioural patterns across online sessions.
- Models trained on the survey data produced unstable results due to the extremely small sample size, highlighting the limitations of self-reported data.

---

## Limitations
- The UCI dataset reflects shopping behaviour from a specific time period (2015–2016).
- The survey dataset is very small and not representative of a broad population.
- No hyperparameter tuning was performed.
- Only a subset of available features was used to keep the analysis interpretable.

---

## Repository Structure

.
├── src/
│ └── main.py
├── data/
│ ├── online_shoppers_intention.csv
│ └── Online Shopping Behaviour 2.csv
├── figures/
│ ├── uci_pca_clusters.png
│ └── survey_pca_clusters.png
├── report.pdf
├── requirements.txt
└── README.md


---

## How to Run
1. Install dependencies:
pip install -r requirements.txt

2. Run the analysis:
python src/main.py


---

## Tech Stack
- Python
- pandas
- NumPy
- scikit-learn
- matplotlib

---

## Future Improvements
- Hyperparameter tuning for supervised and unsupervised models
- Feature importance analysis and cluster profiling
- Larger and more diverse survey data collection
- Evaluation using ROC curves and precision-recall metrics
