import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import sys


# load the CSV file - you can pass the path as argument or just change it here
if len(sys.argv) > 1:
    csv_path = sys.argv[1]
else:
    csv_path = "cs2_stats.csv"

print(f"Loading data from: {csv_path}")
df = pd.read_csv(csv_path)
print("Done.\n")

# basic info about the dataset
print("=== Dataset Info ===")
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print()
print("Columns and types:")
for col in df.columns:
    print(f"  {col}: {df[col].dtype}")
print()

# descriptive stats
print("=== Descriptive Statistics ===")
print(df.describe())
print()

# check for missing values
print("=== Missing Values ===")
for col in df.columns:
    missing = df[col].isnull().sum()
    if missing > 0:
        print(f"  {col}: {missing} missing")
if df.isnull().sum().sum() == 0:
    print("  No missing values found.")
print()

# --- PLOTS ---

# scatter plot: kills vs deaths
print("Creating scatter plot (kills vs deaths)...")
plt.figure(figsize=(8, 5))
plt.scatter(df["kills"], df["deaths"], alpha=0.5, color="tomato")
plt.xlabel("Kills")
plt.ylabel("Deaths")
plt.title("Kills vs Deaths")
plt.tight_layout()
plt.savefig("kills_vs_deaths.png")
print("Saved kills_vs_deaths.png")
plt.show()

# histogram of ADR
print("Creating ADR histogram...")
plt.figure(figsize=(7, 4))
plt.hist(df["adr"], bins=30, color="steelblue", edgecolor="black")
plt.xlabel("ADR (Average Damage per Round)")
plt.ylabel("Frequency")
plt.title("ADR Distribution")
plt.tight_layout()
plt.savefig("adr_histogram.png")
print("Saved adr_histogram.png")
plt.show()

# correlation heatmap
print("Creating correlation heatmap...")
plt.figure(figsize=(10, 7))
# only use numeric columns for correlation
numeric_df = df.select_dtypes(include="number")
corr_matrix = numeric_df.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
print("Saved correlation_heatmap.png")
plt.show()

# --- MACHINE LEARNING ---
print("\n=== Random Forest Classification ===")

# assume "top_performer" is the target column (1 = top, 0 = not top)
target_col = "top_performer"

# separate features and target
X = df.select_dtypes(include="number").drop(columns=[target_col])
y = df[target_col]

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")
print()

# train the model
print("Training RandomForest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Training done.\n")

# make predictions
y_pred = model.predict(X_test)

# print results
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print()
print("Classification Report:")
print(classification_report(y_test, y_pred))

# feature importance
print("Top 5 most important features:")
importances = model.feature_importances_
feature_names = X.columns.tolist()

importance_pairs = []
for i in range(len(feature_names)):
    importance_pairs.append((feature_names[i], importances[i]))

# sort by importance (basic bubble sort style)
importance_pairs.sort(key=lambda x: x[1], reverse=True)

for i in range(min(5, len(importance_pairs))):
    name, score = importance_pairs[i]
    print(f"  {i+1}. {name}: {score:.4f}")
