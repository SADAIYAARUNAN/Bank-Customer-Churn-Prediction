import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    df = pd.read_csv('bank_customer_data.csv')
except:
    print("❌ Error: Run generate_data.py first!")
    exit()

print("\n--- 🏦 BANK CHURN REPORT ---")

# 2. Overall Churn Rate
total_churn = df['Exited'].mean() * 100
print(f"Overall Churn Rate: {total_churn:.2f}%")

# 3. Churn by Country (The "Germany" Question)
print("\n--- 🌍 Churn Rate by Country ---")
churn_by_country = df.groupby('Geography')['Exited'].mean() * 100
print(churn_by_country)

# 4. Churn by Gender
print("\n--- 👫 Churn Rate by Gender ---")
print(df.groupby('Gender')['Exited'].mean() * 100)

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='CreditScore', hue='Exited', multiple='stack')
plt.title('Credit Score vs Churn (Low Scores Leave)')
plt.savefig('churn_graph.png')
print("\n✅ Graph saved as 'churn_graph.png'")