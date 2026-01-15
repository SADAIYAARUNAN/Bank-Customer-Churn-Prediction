import pandas as pd
import numpy as np

np.random.seed(42)

n_customers = 1000

print("----generating bank data----")

data = {
    'CustomerID': range(10001, 10001 + n_customers),
    'CreditScore': np.random.randint(350, 850, n_customers),
    'Geography': np.random.choice(['France', 'Germany', 'Spain'], n_customers),
    'Gender': np.random.choice(['Male', 'Female'], n_customers),
    'Age': np.random.randint(18, 92, n_customers),
    'Tenure': np.random.randint(0, 10, n_customers),
    'Balance': np.random.uniform(0, 250000, n_customers).round(2),
    'NumOfProducts': np.random.choice([1, 2, 3, 4], n_customers),
    'HasCrCard': np.random.choice([0, 1], n_customers),
    'IsActiveMember': np.random.choice([0, 1], n_customers),
    'EstimatedSalary': np.random.uniform(10000, 200000, n_customers).round(2),
}

df = pd.DataFrame(data)

def generate_exit(row):
    score = 0;
    if row['Geography'] == 'Germany' : score += 0.3
    if row['Age'] > 50 : score += 0.4
    if row['CreditScore'] < 50: score += 0.3
    if row['NumOfProducts'] > 2 :score += 0.4
    if row['IsActiveMember'] == 1: score -= 0.3

    probability = score + np.random.normal(0, 0.1)
    return 1 if probability > 0.5 else 0

df['Exited'] = df.apply(generate_exit,axis=1)

df.to_csv('bank_customer_data.csv',index=False)
print(f"✅ Success! Created 'bank_customer_data.csv' with {len(df)} rows.")