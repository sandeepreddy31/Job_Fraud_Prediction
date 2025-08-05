# Job Fraud Prediction

A machine learning project built using Streamlit to detect fraudulent job postings. This project leverages a dataset with real-world job listings and aims to enhance trust and transparency in the job market.

---

## Problem Statement

Online job boards are increasingly plagued by fake or fraudulent job postings, which exploit job seekers and damage the credibility of platforms. This project aims to build a model that can distinguish between genuine and fraudulent job posts using NLP and ML techniques.

---

## Dataset

* Source: [Kaggle - Fake Job Postings](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
* Features: `title`, `location`, `department`, `company_profile`, `description`, `requirements`, `benefits`, `telecommuting`, `employment_type`, `required_experience`, etc.
* Target: `fraudulent` (0: Genuine, 1: Fraudulent)

---

## Tech Stack

* **Python**
* **Pandas, NumPy, Scikit-learn** for data preprocessing and modeling
* **Streamlit** for UI
* **Matplotlib, Seaborn** for visualization
* **Git & GitHub** for version control and collaboration

---

## Model Performance

```text
Classification Report:
               precision    recall  f1-score   support

           0       0.97      1.00      0.98      1693
           1       1.00      0.45      0.62        95

    accuracy                           0.97      1788
   macro avg       0.99      0.73      0.80      1788
weighted avg       0.97      0.97      0.97      1788
```

* The model achieves **high overall accuracy (97%)** with excellent precision for fraudulent jobs (class 1), but recall for class 1 is low (45%), meaning many fake jobs are missed.
* To improve fraud detection, focus on **boosting recall** for class 1—possibly through **class balancing, threshold tuning, or anomaly detection techniques.**

---

## Streamlit UI Features

* Upload and analyze new job postings
* View real-time predictions
* Explore visual analytics

---

## How to Run

1. Clone the repo:

```bash
git clone https://github.com/sandeepreddy31/Job_Fraud_Prediction.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

---

## Sample Inputs

```python
{
  "title": "Software Engineer",
  "location": "US, CA, San Francisco",
  "company_profile": "A fast-growing AI startup",
  "description": "Develop and deploy ML systems",
  "requirements": "Python, ML, NLP, cloud experience",
  "benefits": "Remote, Health Insurance, Stock Options",
  "telecommuting": 1,
  "employment_type": "Full-time",
  "required_experience": "Mid-Senior level",
  "required_education": "Bachelor's Degree"
}
```

---

## Project Structure

```
├── app.py                  # Streamlit UI
├── logistic_model.pkl              
├── tfidf_vectorizer.pkl      
├── job-posting-prediction      
├── requirements.txt       # Python dependencies
├── README.md
```

---

## Author

**Satti Sandeep Reddy**
[GitHub](https://github.com/sandeepreddy31)

---

## License

This project is licensed under the MIT License.

---
