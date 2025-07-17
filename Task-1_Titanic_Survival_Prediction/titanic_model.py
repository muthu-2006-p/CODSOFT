
# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc
)

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
# Load and Clean Data
df = pd.read_csv('Titanic-Dataset.csv')
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
# Prepare Features and Target
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = df['Survived']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
# Evaluation Function (with 4 Visualizations)
def evaluate_model(model, name, is_regression=False):
    print(f"\n🔎 {name} Results:")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if is_regression:
        y_pred_prob = y_pred
        y_pred = np.where(y_pred >= 0.5, 1, 0)
    else:
        if hasattr(model, "predict_proba"):
            y_pred_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_prob = y_pred  

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # 1. Confusion Matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    # 2. Actual vs Predicted Histogram
    plt.figure(figsize=(6, 4))
    sns.histplot(y_test - y_pred, bins=3, kde=False)
    plt.title(f"{name} - Actual vs Predicted Difference")
    plt.xlabel("Actual - Predicted")
    plt.ylabel("Count")
    plt.show()
    # 3. Feature Coefficients / Importance
    if hasattr(model, "coef_"):
        plt.figure(figsize=(8, 4))
        sns.barplot(x=X.columns, y=model.coef_[0])
        plt.title(f"{name} - Feature Coefficients")
        plt.xticks(rotation=45)
        plt.show()
    elif hasattr(model, "feature_importances_"):
        plt.figure(figsize=(8, 4))
        sns.barplot(x=X.columns, y=model.feature_importances_)
        plt.title(f"{name} - Feature Importance")
        plt.xticks(rotation=45)
        plt.show()
    else:
        print("No feature importance for this model.")
    if isinstance(model, DecisionTreeClassifier):
       plt.figure(figsize=(16, 8))
       plot_tree(
        model,
        feature_names=X.columns,
        class_names=['Not Survived', 'Survived'],
        filled=True,
        rounded=True,
        fontsize=9
        )
       plt.title(f"{name} - Decision Tree Structure")
       plt.show()
    # 4. ROC Curve
    if isinstance(y_pred_prob, (np.ndarray, list)):
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc(fpr, tpr):.2f})")
        plt.plot([0, 1], [0, 1], '--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{name} - ROC Curve")
        plt.legend()
        plt.show()
# Run Models

# 1. Logistic Regression
evaluate_model(LogisticRegression(), "Logistic Regression")

# 2. Linear Regression (used for classification)
evaluate_model(LinearRegression(), "Linear Regression (used as Classifier)", is_regression=True)

# 3. K-Nearest Neighbors
evaluate_model(KNeighborsClassifier(n_neighbors=5), "K-Nearest Neighbors")

# 4. Decision Tree Classifier
evaluate_model(DecisionTreeClassifier(), "Decision Tree Classifier")
