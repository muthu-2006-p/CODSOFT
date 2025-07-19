 💳 Credit Card Fraud Detection using Machine Learning

This project builds a classification model to detect fraudulent credit card transactions using machine learning techniques. The dataset is highly imbalanced, making fraud detection a challenging and critical task.

 📊 Dataset

- Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Contains transactions made by European cardholders in September 2013.
- Features are anonymized using PCA (except `Time` and `Amount`)
- Target variable: `Class`
  - 0 → Genuine
  - 1 → Fraudulent

 🛠️ Technologies Used

- Python
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn

 📌 Algorithms Used

1. Logistic Regression
2. Random Forest Classifier

Both algorithms are evaluated using:
- Confusion Matrix
- ROC-AUC Score
- Classification Report (Precision, Recall, F1-Score)

 📈 Visualizations

- Heatmaps of Confusion Matrices
- ROC Curve Comparison of both models

 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
