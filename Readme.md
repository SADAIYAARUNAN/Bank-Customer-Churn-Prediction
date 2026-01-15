# 🏦 Bank Customer Churn Prediction (AI Risk Detector)

### 🚀 Business Problem
The bank was facing a significant issue with customer attrition (churn), particularly in the German market. Losing a customer costs the bank approximately **€1,000** in acquisition costs and lost revenue. The goal was to build an intelligent system to flag high-risk customers *before* they leave.

### 💡 The Solution
I built an end-to-end Machine Learning application that:
1.  **Analyzes** historical customer data to find patterns.
2.  **Predicts** the probability of a specific customer leaving (0-100%).
3.  **Visualizes** the key reasons for the risk (e.g., Number of Products, Age) to help managers take action.

### 📊 Key Results
* **Model Accuracy:** 94% (Random Forest Classifier)
* **Precision:** 100% for High-Risk customers (Zero False Alarms).
* **Business Impact:** Enables targeted retention campaigns (e.g., Loyalty Bonuses) for high-risk clients, potentially saving thousands in revenue.

### 🛠️ Tech Stack
* **Language:** Python 3.9
* **Machine Learning:** Scikit-Learn (Random Forest, Pipeline, OneHotEncoder)
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Web App / UI:** Streamlit

### 🖥️ How to Run This Project
1.  **Install Requirements:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn streamlit
    ```

2.  **Generate Data (Simulation):**
    ```bash
    python3 generate_data.py
    ```

3.  **Train the Model:**
    ```bash
    python3 train_model.py
    ```

4.  **Launch the Dashboard:**
    ```bash
    streamlit run churn_app.py
    ```

---
*Created by SADAIYA ARUNAN - Data Science Portfolio*