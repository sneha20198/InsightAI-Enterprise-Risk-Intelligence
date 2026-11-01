import pandas as pd

# Load the generated dataset
df = pd.read_csv("data/ai_use_cases.csv")

print("=" * 70)
print("ENTERPRISE AI RISK ANALYSIS")
print("=" * 70)

# 1. Dataset size
print("\n1. DATASET OVERVIEW")
print("-" * 40)
print("Total AI use cases:", len(df))
print("Total industries:", df["industry"].nunique())

# 2. Risk distribution
print("\n2. RISK DISTRIBUTION")
print("-" * 40)
print(df["risk_level"].value_counts())

# 3. Average risk by industry
print("\n3. AVERAGE RISK BY INDUSTRY")
print("-" * 40)

industry_risk = (
    df.groupby("industry")["risk_score"]
    .mean()
    .sort_values(ascending=False)
)

print(industry_risk.round(2))

# 4. Highest-risk AI use cases
print("\n4. TOP 10 HIGHEST-RISK AI USE CASES")
print("-" * 40)

top_risk = (
    df[
        [
            "use_case_name",
            "industry",
            "risk_score",
            "risk_level"
        ]
    ]
    .sort_values("risk_score", ascending=False)
    .head(10)
)

print(top_risk.to_string(index=False))

# 5. High-risk percentage
high_risk_percentage = (
    (df["risk_level"] == "High").mean() * 100
)

print("\n5. HIGH-RISK PORTFOLIO")
print("-" * 40)
print(f"High-risk AI systems: {high_risk_percentage:.2f}%")

# 6. Average risk factors
print("\n6. AVERAGE RISK FACTORS")
print("-" * 40)

risk_factors = [
    "data_sensitivity",
    "automation_level",
    "human_oversight",
    "regulatory_exposure",
    "model_transparency",
    "business_criticality",
    "bias_risk",
    "error_rate"
]

print(df[risk_factors].mean().round(2))

# 7. High-risk use cases by industry
print("\n7. HIGH-RISK USE CASES BY INDUSTRY")
print("-" * 40)

high_risk_by_industry = (
    df[df["risk_level"] == "High"]
    .groupby("industry")
    .size()
    .sort_values(ascending=False)
)

print(high_risk_by_industry)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)