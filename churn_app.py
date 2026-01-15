import streamlit as st
import pandas as pd
import joblib

# 1. Load the Brain
model = joblib.load('churn_model.pkl')

st.title("🏦 Customer Churn Risk Detector")
st.write("Enter customer details to predict the risk of them leaving the bank.")

# 2. Create the Input Form (2 Columns for better layout)
col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input("Credit Score", 300, 850, 600)
    geography = st.selectbox("Country", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 92, 40)
    tenure = st.slider("Tenure (Years with Bank)", 0, 10, 3)

with col2:
    balance = st.number_input("Account Balance (€)", 0.0, 250000.0, 60000.0)
    products = st.slider("Number of Products", 1, 4, 2)
    has_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
    is_active = st.selectbox("Is Active Member?", ["Yes", "No"])
    salary = st.number_input("Estimated Salary (€)", 0.0, 200000.0, 50000.0)

# Convert "Yes/No" to 1/0 for the AI
card_code = 1 if has_card == "Yes" else 0
active_code = 1 if is_active == "Yes" else 0

# 3. The Prediction Logic
if st.button("Analyze Risk"):
    # Prepare the data exactly how the model expects it
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Geography': [geography],
        'Gender': [gender],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [products],
        'HasCrCard': [card_code],
        'IsActiveMember': [active_code],
        'EstimatedSalary': [salary]
    })
    
    # Get Prediction (0 or 1) AND Probability (% Chance)
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] # Probability of "1" (Leaving)
    
    # 4. Display Result
    st.markdown("---")
    if probability > 0.5:
        st.error(f"⚠️ HIGH RISK ALERT")
        st.write(f"This customer has a **{probability:.1%}** chance of leaving.")
        st.write("Recommendation: **Offer a loyalty bonus immediately.**")
    else:
        st.success(f"✅ SAFE CUSTOMER")
        st.write(f"Risk Score: **{probability:.1%}** (Low)")


# ... previous code ends here ...

    # 5. Explain WHY (Feature Importance)
    st.subheader("📊 Why this prediction?")
    
    # Extract the importance numbers from the random forest
    # Access the 'classifier' step inside the pipeline
    importance = model.named_steps['classifier'].feature_importances_
    
    # Match the numbers to the names
    # (Note: This is a simplified view. Real OneHotEncoding makes this complex, 
    # so we will manually list the main original features for a clean chart)
    feature_names = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
                     'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Salary']
    
    # Create a DataFrame for the chart
    # We take the first 10 importances (approximate for this demo)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance[:10] 
    }).sort_values(by='Importance', ascending=False)

    # Draw the Bar Chart
    st.bar_chart(importance_df.set_index('Feature'))