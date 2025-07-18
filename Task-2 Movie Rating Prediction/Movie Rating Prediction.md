Movie Rating Prediction with Python
📌 Project Description
This project predicts the IMDb rating of Indian movies based on features such as Genre, Director, and Lead Actors using multiple machine learning algorithms. It aims to understand which features influence movie ratings and build a model to accurately estimate them.
🧠 Machine Learning Models Used
🔹 Linear Regression

🔹 Support Vector Regression (SVR)

🔹 Random Forest Regressor

🔹 K-Nearest Neighbors Regressor (KNN)

🔹 AdaBoost Regressor

Each model is trained and evaluated with performance metrics and a visualization comparing actual vs predicted ratings.

🧰 Tech Stack
Language: Python

Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn📁 Dataset
File: IMDb Movies India.csv

Features used:

Genre

Director

Actor 1, Actor 2, Actor 3

Rating (Target)
📊 Evaluation Metrics
R² Score: Measures the proportion of variance explained

RMSE (Root Mean Squared Error): Measures prediction error

A scatter plot for each model visualizes Actual vs Predicted Ratings.
| Model             | R² Score | RMSE |
| ----------------- | -------- | ---- |
| Random Forest     | 0.72     | 0.65 |
| AdaBoost          | 0.65     | 0.72 |
| Linear Regression | 0.58     | 0.79 |
| KNN Regressor     | 0.55     | 0.82 |
| SVM (SVR)         | 0.47     | 0.90 |

✅ Conclusion
Random Forest generally provides the best prediction accuracy.

The project demonstrates preprocessing, feature encoding, model comparison, and evaluation with Python.

