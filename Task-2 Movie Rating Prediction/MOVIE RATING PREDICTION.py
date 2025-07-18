# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load Dataset
df = pd.read_csv("/content/IMDb Movies India.csv", encoding="ISO-8859-1")

# Select useful columns & drop missing
df = df[['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Rating']].dropna()

# Encode categorical features
le = LabelEncoder()
for col in ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']:
    df[col] = le.fit_transform(df[col].astype(str))

# Define Features & Target
X = df.drop("Rating", axis=1)
y = df["Rating"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Models
models = {
    "Linear Regression": LinearRegression(),
    "SVM (SVR)": SVR(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "KNN Regressor": KNeighborsRegressor(n_neighbors=5),
    "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42)
}

# Train, Predict, Evaluate, Visualize
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[name] = {"R2 Score": round(r2, 4), "RMSE": round(rmse, 4)}

    # Visualization
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.xlabel("Actual Rating")
    plt.ylabel("Predicted Rating")
    plt.title(f"{name} - Actual vs Predicted")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Summary Table
results_df = pd.DataFrame(results).T.sort_values(by="R2 Score", ascending=False)
print("\nPerformance Comparison of Algorithms:\n")
print(results_df)
