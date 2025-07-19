# ----------------------------------------
# Sales Prediction Using 4 ML Algorithms
# ----------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("/content/advertising.csv")

# ----------------------------------------
# Data Preprocessing
# ----------------------------------------
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------------------
# Define Models
# ----------------------------------------
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Support Vector Regressor': SVR(kernel='linear')
}

# ----------------------------------------
# Train and Evaluate Models
# ----------------------------------------
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Evaluation metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    results[name] = {'MSE': mse, 'R2': r2}

    # Plot: Actual vs Predicted
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, predictions, color='blue', alpha=0.7)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title(f"{name} - Actual vs Predicted")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{name.replace(' ', '_')}_prediction_plot.png")
    plt.show()

# ----------------------------------------
# Print Results
# ----------------------------------------
print("\n--- Model Evaluation Summary ---")
for name, metrics in results.items():
    print(f"{name}: MSE = {metrics['MSE']:.2f}, R² = {metrics['R2']:.4f}")
