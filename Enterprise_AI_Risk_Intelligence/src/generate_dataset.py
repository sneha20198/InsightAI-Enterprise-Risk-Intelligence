import pandas as pd
import numpy as np

np.random.seed(42)

use_cases = [
    ("AI Resume Screening", "Human Resources", "Recruitment"),
    ("Loan Approval AI", "Banking", "Credit"),
    ("Fraud Detection AI", "Banking", "Fraud"),
    ("Customer Support Chatbot", "Retail", "Customer Service"),
    ("Medical Diagnosis AI", "Healthcare", "Diagnosis"),
    ("Insurance Claim Assessment", "Insurance", "Claims"),
    ("Product Recommendation AI", "E-commerce", "Marketing"),
    ("Employee Performance Prediction", "Human Resources", "Performance"),
    ("Predictive Maintenance AI", "Manufacturing", "Operations"),
    ("Credit Risk Assessment", "Banking", "Risk"),
    ("Customer Churn Prediction", "Telecom", "Customer Analytics"),
    ("Demand Forecasting AI", "Retail", "Supply Chain"),
    ("Cyber Threat Detection AI", "Technology", "Cybersecurity"),
    ("Invoice Processing AI", "Finance", "Finance Operations"),
    ("Quality Inspection AI", "Manufacturing", "Quality Control"),
    ("Patient Risk Prediction", "Healthcare", "Patient Care"),
    ("Pricing Optimization AI", "Retail", "Pricing"),
    ("Marketing Personalization AI", "E-commerce", "Marketing"),
    ("Legal Document Analysis AI", "Legal", "Legal Operations"),
    ("Supply Chain Risk Prediction", "Logistics", "Supply Chain")
]

records = []

for i in range(1, 301):

    use_case, industry, business_function = use_cases[
        np.random.randint(len(use_cases))
    ]

    data_sensitivity = np.random.randint(1, 4)
    automation_level = np.random.randint(1, 4)

    # 1 = strong human oversight, 3 = no oversight
    human_oversight = np.random.randint(1, 4)

    regulatory_exposure = np.random.randint(1, 4)

    # 1 = highly transparent, 3 = low transparency
    model_transparency = np.random.randint(1, 4)

    business_criticality = np.random.randint(1, 4)

    error_rate = round(np.random.uniform(0.01, 0.20), 3)

    bias_risk = np.random.randint(1, 4)

    records.append({
        "use_case_id": f"AI{i:04d}",
        "use_case_name": use_case,
        "industry": industry,
        "business_function": business_function,
        "data_sensitivity": data_sensitivity,
        "automation_level": automation_level,
        "human_oversight": human_oversight,
        "regulatory_exposure": regulatory_exposure,
        "model_transparency": model_transparency,
        "business_criticality": business_criticality,
        "error_rate": error_rate,
        "bias_risk": bias_risk
    })

df = pd.DataFrame(records)

# Preliminary risk score
df["risk_score"] = (
    df["data_sensitivity"] * 15
    + df["automation_level"] * 10
    + df["human_oversight"] * 10
    + df["regulatory_exposure"] * 15
    + df["model_transparency"] * 10
    + df["business_criticality"] * 15
    + df["bias_risk"] * 10
    + df["error_rate"] * 75
)

# Risk classification
df["risk_level"] = pd.cut(
    df["risk_score"],
    bins=[-np.inf, 110, 165, np.inf],
    labels=["Low", "Medium", "High"]
)

# Save dataset
df.to_csv("data/ai_use_cases.csv", index=False)

print("=" * 60)
print("ENTERPRISE AI RISK DATASET CREATED")
print("=" * 60)

print(f"\nTotal records: {len(df)}")

print("\nRisk distribution:")
print(df["risk_level"].value_counts())

print("\nIndustries:")
print(df["industry"].value_counts())

print("\nAverage risk score:")
print(round(df["risk_score"].mean(), 2))

print("\nSample records:")
print(df.head(5).to_string(index=False))

print("\nSaved to:")
print("data/ai_use_cases.csv")