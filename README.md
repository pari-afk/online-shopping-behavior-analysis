# Online Shopping Behaviour Analysis

## Overview
This project analyzes online shopping behaviour using both real-world clickstream data and self-reported survey responses. The primary objective is to predict whether an online shopping session will result in a purchase and to explore differences between objective behavioural data and how shoppers perceive their own behaviour.

Two datasets were used:
- The UCI Online Shoppers Purchasing Intention dataset (12,000+ sessions)
- A small, self-collected survey dataset (n = 20)

By comparing these datasets, the project examines potential mismatches between shopping intention and actual behaviour.

---
## Proof of Work

**Question:**  
Can online shopping purchase behavior be predicted from session-level clickstream data, and how does this compare to how shoppers perceive their own behavior?

**What I Built:**  
- Binary classification models (Logistic Regression, Decision Tree) to predict purchase intent  
- K-Means clustering and PCA to uncover behavioral session patterns  
- Comparative analysis between objective clickstream data and self-reported survey responses

**Key Results:**  
- Logistic Regression achieved ~**88% accuracy** on the UCI dataset  
- Decision Tree achieved ~**85% accuracy**, highlighting different error trade-offs  
- PCA + clustering revealed distinct browsing and purchasing behavior groups  
- Survey results showed noticeable gaps between perceived and actual shopping behavior

**Artifacts:**  
- PCA cluster visualizations  
- Confusion matrices and model evaluation outputs  

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

