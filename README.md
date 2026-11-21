# Student Lifestyle and Stress Level Prediction

A machine learning project that predicts whether a student is stressed from lifestyle and academic factors, and finds out which factors matter most.

## Dataset

`student-lifestyle-and-stress-dataset.csv` has 25,500 students with these columns:

| Column | Description |
|---|---|
| Student_Type | school, college or working_student |
| Sleep_Hours | Average hours of sleep |
| Study_Hours | Hours spent studying |
| Social_Media_Hours | Hours on social media |
| Attendance | Attendance percentage |
| Exam_Pressure | Exam pressure score (1-10) |
| Family_Support | Family support score (1-10) |
| Month | Month of the year (1-12) |
| Stress_Level | **Target:** 1 = stressed, 0 = not stressed (about 30% stressed) |

## What the project does

1. **Data cleaning:** removes duplicate rows, marks impossible values (negative or very high study hours, attendance outside 1-100) as missing, and fills missing values inside the modeling pipeline so no information leaks from the test set.
2. **Exploratory analysis:** target balance, stress rate by student type and month, feature distributions, correlation heatmap.
3. **Modeling:** Logistic Regression, Decision Tree, Random Forest and Gradient Boosting, compared with 5-fold cross-validation and a held-out test set. Balanced class weights handle the 70/30 class imbalance.
4. **Interpretation:** permutation importance and logistic regression coefficients show which factors drive stress.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Gradient Boosting | 0.797 | 0.635 | 0.763 | 0.693 | 0.869 |
| Logistic Regression | 0.789 | 0.617 | 0.783 | 0.690 | 0.868 |
| Random Forest | 0.800 | 0.645 | 0.745 | 0.691 | 0.865 |
| Decision Tree | 0.779 | 0.607 | 0.745 | 0.669 | 0.842 |

**Key findings**

- Exam pressure is by far the strongest driver of stress, followed by study hours.
- Sleep and family support lower stress; social media time adds a little.
- Working students are stressed most often (about 38%), then college students (31%) and school students (20%), but student type adds almost nothing to the model once study hours are known.
- The data shows association, not cause.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook Student_Stress_Prediction.ipynb
```

Then choose **Run → Run All Cells**. Keep the CSV file in the same folder as the notebook.

## Project structure

```
.
├── Student_Stress_Prediction.ipynb              # full analysis and modeling
├── student-lifestyle-and-stress-dataset.csv     # data
├── figures/                                     # saved charts
├── requirements.txt
└── README.md
```

## Tools

Python, pandas, NumPy, matplotlib, seaborn, scikit-learn.
