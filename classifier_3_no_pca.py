import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from IPython.display import display

# Load dataset
df = pd.read_csv("data/class_3_full/full_3.csv")

# Display dataset info
df.info()
print(df.shape)

# Define feature and label columns
X, y = df.iloc[:, :-1], df.iloc[:, -1]

# Normalize features
X_scaled = StandardScaler().fit_transform(X)

# Apply SMOTE to handle class imbalance
X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X_scaled, y)


# Function to plot class distribution
def plot_class_distribution(y, title):
    class_counts = y.value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Pastel1.colors)
    plt.title(title)
    plt.show()


# Plot class distribution before and after SMOTE
plot_class_distribution(y, "Class Distribution Before SMOTE")
plot_class_distribution(pd.Series(y_resampled), "Class Distribution After SMOTE")


# Compute correlation matrix
corr_matrix = df.corr()

# Mask upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Plot the heatmap
plt.figure(figsize=(20, 16))
sns.heatmap(corr_matrix, mask=mask, cmap="coolwarm", annot=False, fmt=".2f",
            linewidths=0.1, cbar=True, square=True)

plt.xticks(rotation=90)
plt.title("Feature Correlation Heatmap")
plt.show()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, stratify=y_resampled,
                                                    random_state=42)


# Function to perform GridSearchCV
def train_model(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


# Hyperparameter search space
param_grids = {
    "Logistic Regression": {"C": [0.01, 0.1, 1, 10], "max_iter": [500, 1000]},
    "Decision Tree": {"max_depth": [5, 10, 20]},
    "Random Forest": {"n_estimators": [50, 100, 200], "max_depth": [10, 20]},
    "K-Nearest Neighbors": {"n_neighbors": [3, 5, 7]},
    "Support Vector Machine": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
    "Neural Network": {"hidden_layer_sizes": [(50,), (100,)], "max_iter": [500, 1000]},
}


# Function to plot confusion matrix
def plot_confusion_matrix(test, pred, name):
    cm = confusion_matrix(test, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - {name}")
    plt.show()


# Train models and evaluate performance
model_stats = []

# Train and evaluate Logistic Regression
print("\nTraining Logistic Regression...")
log_reg = train_model(LogisticRegression(multi_class='multinomial', solver='lbfgs'), param_grids["Logistic Regression"],
                      X_train, y_train)
y_pred = log_reg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
plot_confusion_matrix(y_test, y_pred, "Logistic Regression")
model_stats.append({"Model": "Logistic Regression", "Accuracy": accuracy, "F1 Score": f1})
print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

# Train and evaluate Decision Tree
print("\nTraining Decision Tree...")
decision_tree = train_model(DecisionTreeClassifier(), param_grids["Decision Tree"], X_train, y_train)
y_pred = decision_tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
plot_confusion_matrix(y_test, y_pred, "Decision Tree")
model_stats.append({"Model": "Decision Tree", "Accuracy": accuracy, "F1 Score": f1})
print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

# Train and evaluate Random Forest
print("\nTraining Random Forest...")
random_forest = train_model(RandomForestClassifier(), param_grids["Random Forest"], X_train, y_train)
y_pred = random_forest.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
plot_confusion_matrix(y_test, y_pred, "Random Forest")
model_stats.append({"Model": "Random Forest", "Accuracy": accuracy, "F1 Score": f1})
print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

# Train and evaluate K-Nearest Neighbors
print("\nTraining K-Nearest Neighbors...")
knn = train_model(KNeighborsClassifier(), param_grids["K-Nearest Neighbors"], X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
plot_confusion_matrix(y_test, y_pred, "K-Nearest Neighbors")
model_stats.append({"Model": "K-Nearest Neighbors", "Accuracy": accuracy, "F1 Score": f1})
print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

# Train and evaluate Support Vector Machine
print("\nTraining Support Vector Machine...")
svm = train_model(SVC(), param_grids["Support Vector Machine"], X_train, y_train)
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
plot_confusion_matrix(y_test, y_pred, "Support Vector Machine")
model_stats.append({"Model": "Support Vector Machine", "Accuracy": accuracy, "F1 Score": f1})
print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

# Train and evaluate Neural Network
print("\nTraining Neural Network...")
mlp = train_model(MLPClassifier(), param_grids["Neural Network"], X_train, y_train)
y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
plot_confusion_matrix(y_test, y_pred, "Neural Network")
model_stats.append({"Model": "Neural Network", "Accuracy": accuracy, "F1 Score": f1})
print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

# Create and display model statistics dataframe
model_stats_df = pd.DataFrame(model_stats)
display(model_stats_df)
