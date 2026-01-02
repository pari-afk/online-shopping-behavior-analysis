import pandas as pd
import numpy as np
import os

# sklearn stuff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

print("Current working directory:", os.getcwd())
print("Data files:", os.listdir("data"))

# loading uci online shoppers dataset 
uci_df = pd.read_csv("data/online_shoppers_intention.csv")

# loading ym survey dataset 
survey_df = pd.read_csv("data/survey_shopping_behavior.csv")

print("UCI dataset preview")
print(uci_df.head())

print("Survey dataset preview")
print(survey_df.head())

# in uci dataset
# target: 'revenue' (whether purchase was made)
uci_df_clean = uci_df.dropna().copy()
uci_df_clean["Revenue"] = uci_df_clean["Revenue"].astype(int)
uci_df_clean["Weekend"] = uci_df_clean["Weekend"].astype(int)

# model
uci_features = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Weekend"
]

X_uci = uci_df_clean[uci_features]
y_uci = uci_df_clean["Revenue"]

# train/test split
X_uci_train, X_uci_test, y_uci_train, y_uci_test = train_test_split(
    X_uci, y_uci, test_size=0.3, random_state=42
)

# scaling for uci models
uci_scaler = StandardScaler()
X_uci_train_scaled = uci_scaler.fit_transform(X_uci_train)
X_uci_test_scaled = uci_scaler.transform(X_uci_test)

print("\nUCI training/testing shapes:")
print(X_uci_train.shape, X_uci_test.shape)


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    print(f"\n=== {name} ===")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("Accuracy:", acc)
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))
    return acc

# uci: logistic regression
uci_log_reg = LogisticRegression(max_iter=1000)
evaluate_model("Logistic Regression (UCI)", uci_log_reg,
               X_uci_train_scaled, y_uci_train,
               X_uci_test_scaled, y_uci_test)

# uci: decision tree
uci_dt = DecisionTreeClassifier(random_state=42)
evaluate_model("Decision Tree (UCI)", uci_dt,
               X_uci_train, y_uci_train,
               X_uci_test, y_uci_test)

# uci: k-means
X_uci_all_scaled = uci_scaler.fit_transform(X_uci)

kmeans_uci = KMeans(n_clusters=3, random_state=42, n_init=10)
uci_clusters = kmeans_uci.fit_predict(X_uci_all_scaled)
uci_df_clean.loc[:, "Cluster"] = uci_clusters

print("\nUCI cluster counts:")
print(uci_df_clean["Cluster"].value_counts())

# uci: pca plot 
import matplotlib.pyplot as plt

pca_uci = PCA(n_components=2)
uci_components = pca_uci.fit_transform(X_uci_all_scaled)

uci_df_clean["PC1"] = uci_components[:, 0]
uci_df_clean["PC2"] = uci_components[:, 1]

# sampling to avoid thoudsands of plots 
uci_sample = uci_df_clean.sample(n=500, random_state=42)

plt.figure(figsize=(7, 5))
plt.scatter(
    uci_sample["PC1"],
    uci_sample["PC2"],
    c=uci_sample["Cluster"],
    cmap="viridis",
    s=20
)
plt.title("UCI: Clusters of Online Sessions (via PCA)")
plt.xlabel("Behaviour Dimension 1")
plt.ylabel("Behaviour Dimension 2")
plt.colorbar(label="Cluster (from K-means)")
plt.grid(True)
plt.show()

# survey data
# renaming survey columns 
cols = survey_df.columns

rename_dict = {
    cols[0]: "Q1_HowOften",
    cols[1]: "Q2_ReturningCustomer",
    cols[2]: "Q3_TimeSpent",
    cols[3]: "Q4_ProductPages",
    cols[4]: "Q5_ComparePrices",
    cols[5]: "Q6_ReadReviews",
    cols[6]: "Q7_WeekendShopper",
    cols[7]: "Q9_LikelihoodPurchase",
    cols[8]: "Q11_CompletedPurchase",
    cols[9]: "Q10_CartAbandon",
    cols[10]: "Q12_ShopperType"
}

survey_df = survey_df.rename(columns=rename_dict)

print("\nRenamed survey columns:")
print(survey_df.columns.tolist())

# mpaping survey answers to numbers
map_Q1 = {
    "Rarely": 1,
    "Occasionally (1–3 times per month)": 2,
    "Regularly (once a week)": 3,
    "Frequently (multiple times a week)": 4,
    "Everyday": 5
}

map_Q2 = {
    "Yes, I often use the same stores": 2,
    "Sometimes, depends on the product": 1,
    "No, I mostly browse different stores everytime": 0,
    "Not sure": 0
}

map_Q3 = {
    "Less than 2 minutes": 1,
    "5-10 minutes": 3,
    "10-20 minutes": 4,
    "More than 20 minutes": 5,
    "2-5 minutes": 2
}

map_Q4 = {
    "1-2 pages": 1,
    "3-5 pages": 2,
    "6-10 pages": 3,
    "11-20 pages": 4,
    "More than 20 pages": 5
}

map_Q5 = {
    "Often": 2,
    "Sometimes": 1,
    "In this economy?? (Always)": 3,
    "Always": 3
}

map_Q6 = {
    "Always": 3,
    "Sometimes": 2,
    "Only for expensive items": 1,
    "Never": 0
}

map_Q7 = {
    "Sometimes": 1,
    "Depends on sales or events": 1,
    "Yes": 2,
    "No": 0
}

map_Q9 = {
    "Very likely": 5,
    "Likely": 4,
    "Neutral": 3,
    "Unlikely": 2,
    "Very unlikely": 1
}

map_Q10 = {
    "Yes, multiple times": 2,
    "Yes, once": 1,
    "No": 0
}

map_Q11 = {
    "Yes": 1,
    "No": 0
}

map_Q12 = {
    "Impulse buyer": 0,
    "Occasional shopper": 3,
    "Window shopper (browses but rarely buys)": 4,
    "Research-heavy shopper (reads and compares everything before buying)": 2,
    "Bargain shopper": 1
}

# mappings
survey_df["Q1_num"] = survey_df["Q1_HowOften"].map(map_Q1)
survey_df["Q2_num"] = survey_df["Q2_ReturningCustomer"].map(map_Q2)
survey_df["Q3_num"] = survey_df["Q3_TimeSpent"].map(map_Q3)
survey_df["Q4_num"] = survey_df["Q4_ProductPages"].map(map_Q4)
survey_df["Q5_num"] = survey_df["Q5_ComparePrices"].map(map_Q5)
survey_df["Q6_num"] = survey_df["Q6_ReadReviews"].map(map_Q6)
survey_df["Q7_num"] = survey_df["Q7_WeekendShopper"].map(map_Q7)
survey_df["Q9_num"] = survey_df["Q9_LikelihoodPurchase"].map(map_Q9)
survey_df["Q11_num"] = survey_df["Q11_CompletedPurchase"].map(map_Q11)
survey_df["Q10_num"] = survey_df["Q10_CartAbandon"].map(map_Q10)
survey_df["Q12_num"] = survey_df["Q12_ShopperType"].map(map_Q12)

# dropping rows to avoid errors
survey_df_num = survey_df.dropna().copy()

print("\nNumeric survey preview:")
print(
    survey_df_num[
        [
            "Q1_num",
            "Q2_num",
            "Q3_num",
            "Q4_num",
            "Q5_num",
            "Q6_num",
            "Q7_num",
            "Q9_num",
            "Q10_num",
            "Q11_num",
            "Q12_num",
        ]
    ].head()
)

print("Survey shape after cleaning:", survey_df_num.shape)

# supervised models for comparison

survey_features = [
    "Q1_num",
    "Q2_num",
    "Q3_num",
    "Q4_num",
    "Q5_num",
    "Q6_num",
    "Q7_num",
    "Q9_num",
    "Q10_num",
]

y_survey = survey_df_num["Q11_num"]
X_survey = survey_df_num[survey_features]

X_survey_train, X_survey_test, y_survey_train, y_survey_test = train_test_split(
    X_survey, y_survey, test_size=0.3, random_state=42
)

survey_scaler = StandardScaler()
X_survey_train_scaled = survey_scaler.fit_transform(X_survey_train)
X_survey_test_scaled = survey_scaler.transform(X_survey_test)

print("\nSurvey training/testing shapes:")
print(X_survey_train.shape, X_survey_test.shape)

# logistic regression on survey 
log_reg_survey = LogisticRegression(max_iter=1000)
evaluate_model(
    "Logistic Regression (Survey)",
    log_reg_survey,
    X_survey_train_scaled,
    y_survey_train,
    X_survey_test_scaled,
    y_survey_test,
)

# decision tree on survey 
dt_survey = DecisionTreeClassifier()
evaluate_model(
    "Decision Tree (Survey)",
    dt_survey,
    X_survey_train,
    y_survey_train,
    X_survey_test,
    y_survey_test,
)

# k-means on survey

X_survey_all = survey_df_num[
    ["Q1_num", "Q2_num", "Q3_num", "Q4_num", "Q5_num", "Q6_num", "Q7_num", "Q9_num", "Q10_num"]
]
X_survey_all_scaled = survey_scaler.fit_transform(X_survey_all)

kmeans_survey = KMeans(n_clusters=3, random_state=42, n_init=10)
survey_clusters = kmeans_survey.fit_predict(X_survey_all_scaled)
survey_df_num.loc[:, "Cluster"] = survey_clusters

print("\nSurvey cluster assignments (shopper type vs cluster):")
print(survey_df_num[["Q12_num", "Cluster"]])

pca_survey = PCA(n_components=2)
survey_components = pca_survey.fit_transform(X_survey_all_scaled)

survey_df_num["PC1"] = survey_components[:, 0]
survey_df_num["PC2"] = survey_components[:, 1]

plt.figure(figsize=(7, 5))
plt.scatter(
    survey_df_num["PC1"],
    survey_df_num["PC2"],
    c=survey_df_num["Cluster"],
    cmap="viridis",
    s=80,
)

plt.title("Survey: Clusters of Online Shoppers (via PCA)")
plt.xlabel("Shopping Behaviour Dimension 1")
plt.ylabel("Shopping Behaviour Dimension 2")
plt.colorbar(label="Shopper Group (from K-means)")
plt.grid(True)
plt.show()
