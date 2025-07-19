# credit_card_fraud_detection.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Load dataset
df = pd.read_csv('/content/creditcard.csv')

# Drop any rows with NaN in 'Class'
df = df.dropna(subset=['Class'])

# Separate features and target
X = df.drop(['Class'], axis=1)
y = df['Class']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# ---------------- Logistic Regression ----------------
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_preds = log_model.predict(X_test)

# Evaluation - Logistic Regression
print("=== Logistic Regression ===")
print(classification_report(y_test, log_preds))
print("ROC AUC Score:", roc_auc_score(y_test, log_model.predict_proba(X_test)[:, 1]))

# Confusion Matrix - Logistic
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, log_preds), annot=True, fmt='d', cmap='Blues')
plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ROC Curve - Logistic
fpr1, tpr1, _ = roc_curve(y_test, log_model.predict_proba(X_test)[:, 1])
plt.plot(fpr1, tpr1, label='Logistic Regression (AUC = %0.2f)' % roc_auc_score(y_test, log_model.predict_proba(X_test)[:, 1]))

# ---------------- Random Forest ----------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# Evaluation - Random Forest
print("\n=== Random Forest ===")
print(classification_report(y_test, rf_preds))
print("ROC AUC Score:", roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1]))

# Confusion Matrix - Random Forest
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, rf_preds), annot=True, fmt='d', cmap='Greens')
plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ROC Curve - Random Forest
fpr2, tpr2, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
plt.plot(fpr2, tpr2, label='Random Forest (AUC = %0.2f)' % roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1]))

# Final ROC Plot
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.show()
