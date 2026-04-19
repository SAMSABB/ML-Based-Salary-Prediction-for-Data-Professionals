# Predicting Data Science Salaries using Machine Learning

## 1. Introduction
This project uses machine learning to predict the salary of data professionals based on features such as programming experience, tools used, and job role, using the 2022 Kaggle ML & Data Science Survey dataset.

## 2. Dataset

### 2.1 Data Source
The dataset used in this project is the 2022 Kaggle Machine Learning and Data Science Survey. It is a mutiple choice question survey with questions relating to many aspects of industry individuals careers IE demographic information, education levels, compensation levels as well as job-related information relating to the use of tools/frameworks and their personal experence around role specific tasks. 
**The dataset has 23,998 respondent rows.**

### 2.2 Features Used

**Age**
- Acts as a proxy for career stage and overall professional experience.
**Country**
- Captures geographic differences in salary due to economic conditions, demand, and cost of living.
**Education Level**
- Higher education levels may be associated with specialised roles and higher earning potential.
**Job Role**
- One of the strongest indicators of salary, as different roles have different pay ranges.
**Coding Experience (Years)**
- Reflects technical expertise and is likely to correlate positively with salary.
**Number of Programming Languages Used**
- Used as a proxy for breadth of technical skills.
**Machine Learning Experience**
- Indicates level of specialisation in ML, which can impact access to higher-paying roles.

### 2.3 Target Variable
**Compensation (Annual Salary Band)**
- The target variable representing respondents’ income range, used for classification.

## 3. Methodology
- Data cleaning (missing values, encoding)
- Feature engineering
- Model training & evaluation

## 4. Models Implemented

The following classification models were trained and evaluated:

- Logistic Regression — used as a baseline model
- Decision Tree — captures non-linear relationships in the data
- K-Nearest Neighbours — instance-based learning approach
- Random Forest (Ensemble) — improves performance through aggregation of multiple decision trees

## 5. Model Evaluation
**Metrics:**
- Accuracy
- Recall + Precision
- Confusion Matrix

### Results Table:
| Model                | Accuracy | Weighted F1 | Class 0 F1 | Class 1 F1 |
| -------------------- | -------- | ----------- | ---------- | ---------- |
| Logistic Regression  | 0.73     | 0.70        | 0.82       | 0.48       |
| K-Nearest Neighbours | 0.72     | 0.70        | 0.80       | 0.52       |
| Decision Tree        | 0.71     | 0.70        | 0.80       | 0.50       |
| Random Forest        | **0.74** | **0.72**    | 0.82       | **0.54**   |

**Best Model**
The Random Forest model achieved the best overall performance, with the highest accuracy (0.74) and weighted F1-score (0.72). It also provided the most balanced performance across both classes.

## 6. Key Observations:
- All models performed significantly better on Class 0 than Class 1, suggesting class imbalance or that higher salary bands are harder to predict.
- Logistic Regression achieved strong recall for Class 0 (0.92), but struggled with Class 1 predictions.
- KNN and Decision Tree showed similar performance, indicating limited gains from non-linear models without ensemble methods.
- Random Forest improved performance across all metrics, demonstrating the benefit of ensemble learning.
- Predicting higher salary bands proved more challenging, likely due to greater variability in career paths and influencing factors.

## 7. How To Run

1. Clone the repository  
2. Install dependencies:
   pip install -r requirements.txt  
3. Run the training script:
   python src/train.py

## 8. Future Improvements
- Improve feature engineering
- Use more advanced models (e.g. XGBoost)
- Categorical features were excluded for simplicity; future work will include encoding

