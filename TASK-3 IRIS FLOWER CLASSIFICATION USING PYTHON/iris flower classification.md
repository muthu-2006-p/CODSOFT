 🌸 IRIS Flower Classification - CodSoft Internship Task-3

 📌 Task Objective
Build a machine learning model to classify iris flowers into Setosa , Versicolor, or Virginica based on petal and sepal measurements using Python and Scikit-learn.

---

 📁 Dataset
- Filename: `IRIS.csv`
- Features:
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
- Target: `species` (Setosa, Versicolor, Virginica)

---

 🔧 Tools & Libraries Used
- Python
- Pandas & Numpy
- Matplotlib & Seaborn
- Scikit-learn

---

 📊 Visualizations
- 🔍 Pairplot for class-wise clustering
- 📊 Heatmap for feature correlation
- 📉 Confusion Matrices for each model

Images saved in `/visuals/` directory.

---

 🤖 ML Models Implemented
| Model                 | Accuracy |
|----------------------|----------|
| Logistic Regression  | 100%     |
| Decision Tree        | 100%     |
| K-Nearest Neighbors  | 100%     |
| Support Vector Machine | 100%   |

---

 🧪 Evaluation
Each model was trained on 80% of the data and tested on 20%. All models achieved perfect accuracy due to the simplicity and separability of the dataset.

---

🧠 How to Run

```bash
pip install pandas matplotlib seaborn scikit-learn
python iris_classification.py
