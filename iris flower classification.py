# Re-import and use original data for plotting
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Load original dataset
df = pd.read_csv("/content/IRIS.csv")

# ----- VISUALIZATION 1: Pairplot -----
plt.figure(figsize=(10, 6))
sns.pairplot(df, hue='species')
plt.suptitle("🔍 Pairplot of Iris Features by Species", y=1.02)
plt.show()

# ----- VISUALIZATION 2: Heatmap -----
plt.figure(figsize=(8, 6))
sns.heatmap(df.drop(columns='species').corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("📊 Feature Correlation Heatmap")
plt.show()

# Encode target labels
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

# Split features & labels
X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC()
}

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    results[name] = {
        "Accuracy": acc,
        "Classification Report": report
    }

# Print model evaluation
for name, result in results.items():
    print(f"\n📌 Model: {name}")
    print(f"✅ Accuracy: {result['Accuracy']:.2f}")
    print("🧾 Classification Report:\n", result["Classification Report"])

# ----- VISUALIZATION 3: Confusion Matrices -----
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for i, (name, model) in enumerate(models.items()):
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test,
        display_labels=le.classes_,
        cmap='Blues',
        ax=axes[i]
    )
    axes[i].set_title(f"{name} - Confusion Matrix")

plt.tight_layout()
plt.show()
