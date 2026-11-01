import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# ==========================================
# MYSQL CONNECTION
# ==========================================

connection_url = URL.create(
    drivername="mysql+pymysql",
    username="ai_project",
    password="AIProject@2026",
    host="localhost",
    port=3306,
    database="enterprise_ai_risk"
)

engine = create_engine(connection_url)

# ==========================================
# LOAD CURRENT DATA FROM MYSQL
# ==========================================

query = """
SELECT
    use_case_id,
    use_case_name,
    industry,
    business_function,
    data_sensitivity,
    automation_level,
    human_oversight,
    regulatory_exposure,
    model_transparency,
    business_criticality,
    error_rate,
    bias_risk,
    risk_score,
    risk_level
FROM ai_use_cases;
"""

df = pd.read_sql(query, engine)

print("=" * 60)
print("ENTERPRISE AI RISK — ML TRAINING")
print("=" * 60)

print(f"\nRecords loaded from MySQL: {len(df)}")

# ==========================================
# FEATURES
# ==========================================

features = [
    "data_sensitivity",
    "automation_level",
    "human_oversight",
    "regulatory_exposure",
    "model_transparency",
    "business_criticality",
    "error_rate",
    "bias_risk"
]

X = df[features]
y = df["risk_level"]

# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")

# ==========================================
# RANDOM FOREST
# ==========================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# ==========================================
# EVALUATION
# ==========================================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n" + "=" * 60)
print(f"MODEL ACCURACY: {accuracy:.2%}")
print("=" * 60)

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE")
print("-" * 40)

for _, row in importance.iterrows():
    print(
        f"{row['feature']:<25} "
        f"{row['importance']:.4f}"
    )

# ==========================================
# SAVE MODEL
# ==========================================

os.makedirs("models", exist_ok=True)

model_path = "models/ai_risk_model.pkl"

joblib.dump(model, model_path)

print("\nModel saved to:")
print(model_path)

print("\nML TRAINING COMPLETE")