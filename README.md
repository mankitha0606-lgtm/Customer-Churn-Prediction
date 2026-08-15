# 📊 Customer Churn Prediction

An end-to-end Machine Learning project that predicts customer churn probability and classifies customers based on their churn risk.

The project combines data analysis, SQL, machine learning, model evaluation, SHAP-based model interpretation, and Streamlit deployment to build a practical customer retention solution.

---

## 🚀 Live Demo

🔗 **Streamlit App:**  
https://customer-churn-prediction-zr2ct4huwpvlahctwj9f3z.streamlit.app/

---

## 🎯 Project Objective

Customer churn is an important business problem for subscription-based companies.

The objective of this project is to:

- Predict whether a customer is likely to churn.
- Estimate the probability of customer churn.
- Categorize customers into Low, Medium, and High Risk.
- Identify important factors influencing churn predictions.
- Provide an interactive interface for real-time prediction.
- Use SHAP to understand individual model predictions.


---

## 📊 Dataset

The project uses customer subscription and service-related information to predict customer churn.

### Important Features

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies
- Contract
- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges

### Target Variable

```text
Churn
```

Possible values:

```text
Yes
No
```

---

## 🔍 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand customer behavior and identify patterns related to churn.

The analysis included:

- Customer demographics
- Service subscriptions
- Contract types
- Payment methods
- Internet services
- Customer tenure
- Monthly charges
- Total charges
- Churn distribution
- Relationships between customer attributes and churn

The EDA helped identify customer segments and features associated with higher churn.

---

## 🗄️ SQL Analysis

SQL was used to perform analytical queries on the customer dataset.

The analysis focused on:

- Overall churn rate
- Churn by contract type
- Churn by payment method
- Churn by internet service
- Customer tenure
- Monthly charges
- Total charges
- Customer service usage

SQL analysis provided additional business insights and supported the Machine Learning workflow.

---

## ⚙️ Data Preprocessing

The dataset contains both numerical and categorical features.

### Numerical Features

Numerical features were standardized using:

```python
StandardScaler()
```

### Categorical Features

Categorical variables were converted into numerical representations using:

```python
OneHotEncoder(
    handle_unknown="ignore",
    drop="first"
)
```

A `ColumnTransformer` was used to apply different preprocessing techniques to numerical and categorical features.

---

## 🤖 Machine Learning Model

A **Random Forest Classifier** was trained to predict customer churn.

The model generates:

1. Churn prediction
2. Churn probability

The predicted probability is then used to classify the customer's churn risk.

### Model

```text
Random Forest Classifier
```

---

## 📈 Model Performance

The Random Forest model achieved the following performance on the test dataset:

| Metric | Score |
|---|---:|
| Accuracy | 77.26% |
| Precision | 56.14% |
| Recall | 66.04% |
| F1 Score | 60.69% |
| ROC-AUC | 82.04% |

The model provides a reasonable balance between identifying potential churners and minimizing incorrect predictions.

---

## 🎯 Probability Threshold Optimization

Instead of using the default classification threshold of `0.50`, a threshold of:

```text
0.40
```

was selected for the final churn prediction.

This allows the model to identify more potential churners by increasing sensitivity toward the positive class.

### Final Prediction Logic

```text
Churn Probability ≥ 0.40
        ↓
      Churn

Churn Probability < 0.40
        ↓
    Will Stay
```

---

## 🚦 Customer Risk Classification

Customers are categorized according to their predicted churn probability.

| Churn Probability | Risk Level |
|---|---|
| `< 40%` | 🟢 Low Risk |
| `40% – 69.99%` | 🟡 Medium Risk |
| `≥ 70%` | 🔴 High Risk |

This provides an easier way for businesses to prioritize customer retention efforts.

---

## 🧠 SHAP Model Interpretation

SHAP (**SHapley Additive exPlanations**) was used to understand how individual features influence the Random Forest predictions.

The SHAP analysis provides:

- Global feature importance
- Individual customer explanations
- Positive feature contributions
- Negative feature contributions
- Feature impact on churn predictions

### Important Predictive Features

The analysis identified several important factors influencing churn predictions, including:

- Tenure
- Contract type
- Internet service
- Total charges
- Monthly charges
- Payment method
- Technical support
- Online security

SHAP makes the Machine Learning model easier to interpret by showing **why a particular prediction was made**.

---

### Application Features

- 👤 Customer profile input
- 📊 Churn probability
- 🎯 Churn prediction
- 🚦 Risk classification
- 🧠 SHAP-based prediction explanation
- 📈 Visual prediction results
- ⚡ Real-time model inference


---

## 🛠️ Technologies Used

### Programming Language

- Python

### Data Analysis

- Pandas
- NumPy
- Matplotlib

### Machine Learning

- Scikit-learn
- Random Forest Classifier
- StandardScaler
- OneHotEncoder
- ColumnTransformer

### Explainable AI

- SHAP

### Database & Analytics

- SQL

### Deployment

- Streamlit
- Git
- GitHub

### Model Serialization

- Joblib

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/mankitha0606-lgtm/Customer-Churn-Prediction.git
```

Navigate to the project directory:

```bash
cd Customer-Churn-Prediction
```

Install the required dependencies:

```bash
pip install -r Model_Deployment/requirements.txt
```

---

## ▶️ Run the Application Locally

Navigate to the deployment folder:

```bash
cd Model_Deployment
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🌍 Deployment

The application is deployed using **Streamlit Community Cloud**.

```

### Live Application

🔗 https://customer-churn-prediction-zr2ct4huwpvlahctwj9f3z.streamlit.app/

---

## 💼 Business Value

The project can help businesses identify customers who may be at risk of leaving.

Potential business applications include:

- Identifying high-risk customers
- Prioritizing retention campaigns
- Understanding major churn drivers
- Targeting customers with personalized offers
- Improving customer retention strategies
- Supporting data-driven business decisions

Instead of treating every customer equally, businesses can prioritize retention efforts based on predicted churn probability.


---

## 📌 Key Project Highlights

- ✅ End-to-end Machine Learning workflow
- ✅ Exploratory Data Analysis
- ✅ SQL-based business analysis
- ✅ Random Forest classification
- ✅ Probability threshold optimization
- ✅ Churn risk classification
- ✅ SHAP model interpretation
- ✅ Interactive Streamlit application
- ✅ Cloud deployment
- ✅ GitHub version control

---

## 👩‍💻 Author

### M Ankitha

**Data Science & Machine Learning Enthusiast**

Interested in Data Analytics, Machine Learning, Artificial Intelligence, and building practical data-driven solutions.

---

## ⭐ Acknowledgements

This project was developed as a practical implementation of Machine Learning, Data Analytics, SQL, Explainable AI, and Model Deployment concepts.

---

## 📜 License

This project is intended for educational and portfolio purposes.
