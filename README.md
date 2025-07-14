## 🪀 Heart Disease Prediction Using Machine Learning

### 📌 Project Overview

This project focuses on developing a machine learning-based system to predict the presence of heart disease in patients using clinical features such as age, chest pain type, cholesterol, resting blood pressure, maximum heart rate, and more. It includes full data preprocessing, model training, hyperparameter tuning, evaluation, and visualization.

---

### 🌟 Objective

To build reliable machine learning models that can predict the likelihood of heart disease in a patient and assist in early diagnosis using supervised classification techniques.

---

### 🗂️ Dataset Description

The dataset includes 303 records with 14 attributes, including:

| Column     | Description                       |
| ---------- | --------------------------------- |
| `age`      | Age of the patient                |
| `sex`      | Gender (0 = Female, 1 = Male)     |
| `cp`       | Chest pain type (0–3)             |
| `trtbps`   | Resting blood pressure            |
| `chol`     | Serum cholesterol                 |
| `fbs`      | Fasting blood sugar (>120 mg/dL)  |
| `restecg`  | Resting ECG results               |
| `thalachh` | Maximum heart rate achieved       |
| `exng`     | Exercise-induced angina           |
| `oldpeak`  | ST depression during exercise     |
| `slp`      | Slope of ST segment               |
| `caa`      | Number of major vessels           |
| `thall`    | Thalassemia type                  |
| `output`   | Target variable (0 = No, 1 = Yes) |

---

### 🔧 Workflow

1. Data Preprocessing
2. Handling Missing Values (Median, Mode, Interpolation)
3. Outlier Detection & Treatment (IQR Method)
4. Feature Scaling using StandardScaler
5. Data Visualization (boxplot, violin, lineplot, pie, heatmap)
6. Model Building:

   * Logistic Regression
   * Decision Tree
   * Random Forest
   * Gradient Boosting
   * XGBoost
   * Support Vector Machine (SVM)
7. Hyperparameter Tuning
8. Evaluation using Accuracy, F1-score, ROC-AUC
9. Sample Predictions
10. Model Deployment using Pickle

---

### 📊 Data Visualization Highlights

* Gender distribution pie chart
* Chest pain vs. heart disease bar plot
* Age distribution of heart disease patients
* Correlation heatmap of features
* Cholesterol and heart rate boxplots
* Exercise-induced angina vs. heart disease violin plot

---

### 🧠 Machine Learning Models & Performance

| Model               | Accuracy | F1-Score | Comments                         |
| ------------------- | -------- | -------- | -------------------------------- |
| Logistic Regression | 86%      | 0.86     | Fast & interpretable             |
| Decision Tree       | 86%      | 0.86     | Good after tuning                |
| Random Forest       | 85%      | 0.85     | Slight overfitting improved      |
| Gradient Boosting   | 85%      | 0.85     | Balanced performance             |
| XGBoost             | 87%      | 0.87     | Best performance overall         |
| SVM                 | 86%      | 0.86     | Strong and stable classification |

---

### 📈 Sample Insights

* Most heart disease cases occur in patients aged **45–60**.
* Males are more affected by heart disease than females.
* Patients without angina tend to have higher heart rates.
* Features like `thalachh` and `caa` are highly correlated with heart disease.

---

### 🧪 Evaluation Metrics

* Accuracy
* Confusion Matrix
* Precision, Recall, F1-score
* ROC-AUC Score
* Cross-Validation Accuracy

---

### 📦 Deployment

The best model (`XGBoost`) and scaler are saved using `pickle`:

```bash
model.pkl
scaler.pkl
```

These can be used in Flask, FastAPI, or any deployment framework for real-time prediction.

---

### ✅ Conclusion

This project demonstrates the power of machine learning in detecting heart disease. Among all models, **XGBoost** achieved the best balance between **accuracy (87%)** and **generalization**, making it an ideal choice for deployment in healthcare applications.
