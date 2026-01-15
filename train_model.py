import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the Evidence
data = pd.read_csv('bank_customer_data.csv')

X = data.drop(['Exited','CustomerID'], axis=1)
y = data['Exited']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_features = ['Geography','Gender']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'),categorical_features)
    ],
    remainder='passthrough'
)


model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 6. Train the Model
print("🤖 Training the AI... (Reading 800 customers)")
model_pipeline.fit(X_train, y_train)

# 7. Test the Model
print("📝 Testing the AI... (Predicting 200 customers)")
y_pred = model_pipeline.predict(X_test)

# 8. Report Card
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.2f}")
print("\n--- Detailed Report ---")
print(classification_report(y_test, y_pred))

# 9. Save the Brain
joblib.dump(model_pipeline, 'churn_model.pkl')
print("💾 Model saved as 'churn_model.pkl'")