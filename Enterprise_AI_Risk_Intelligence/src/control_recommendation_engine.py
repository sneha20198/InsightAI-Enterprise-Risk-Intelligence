import pandas as pd

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("data/ai_use_cases.csv")

system = df.iloc[0]

# ============================================================
# CONTROL LIBRARY
# ============================================================

control_library = {

    "regulatory_exposure": {
        "name": "Regulatory Compliance Review",
        "action": "Conduct regulatory assessment and document applicable obligations.",
        "owner": "Compliance / Legal",
        "priority": "High"
    },

    "business_criticality": {
        "name": "Human Approval Gate",
        "action": "Restrict autonomous decisions and require human approval for high-impact outcomes.",
        "owner": "Business Owner",
        "priority": "Critical"
    },

    "human_oversight": {
        "name": "Human-in-the-Loop Control",
        "action": "Introduce mandatory human review for sensitive or high-impact decisions.",
        "owner": "AI Governance",
        "priority": "High"
    },

    "data_sensitivity": {
        "name": "Data Protection Control",
        "action": "Apply stronger access controls, minimization and data protection measures.",
        "owner": "Data Governance",
        "priority": "High"
    },

    "bias_risk": {
        "name": "Fairness Assessment",
        "action": "Perform bias testing across relevant demographic or business segments.",
        "owner": "Responsible AI",
        "priority": "High"
    },

    "model_transparency": {
        "name": "Explainability Documentation",
        "action": "Document model logic, limitations and explanation methodology.",
        "owner": "Model Risk",
        "priority": "Medium"
    },

    "automation_level": {
        "name": "Automation Boundary",
        "action": "Limit autonomous decision authority and introduce escalation thresholds.",
        "owner": "Business Owner",
        "priority": "High"
    },

    "error_rate": {
        "name": "Model Monitoring",
        "action": "Implement continuous performance monitoring and model validation.",
        "owner": "ML Engineering",
        "priority": "High"
    }
}

# ============================================================
# IDENTIFY CONTROL GAPS
# ============================================================

gaps = []

if system["regulatory_exposure"] >= 3:
    gaps.append("regulatory_exposure")

if system["business_criticality"] >= 3:
    gaps.append("business_criticality")

if system["human_oversight"] >= 2:
    gaps.append("human_oversight")

if system["data_sensitivity"] >= 3:
    gaps.append("data_sensitivity")

if system["bias_risk"] >= 2:
    gaps.append("bias_risk")

if system["model_transparency"] >= 2:
    gaps.append("model_transparency")

if system["automation_level"] >= 3:
    gaps.append("automation_level")

if system["error_rate"] >= 0.10:
    gaps.append("error_rate")

# ============================================================
# BUILD CONTROL PLAN
# ============================================================

recommendations = []

for gap in gaps:

    control = control_library[gap]

    recommendations.append({

        "risk_driver": gap,

        "control": control["name"],

        "recommended_action": control["action"],

        "owner": control["owner"],

        "priority": control["priority"]
    })

control_plan = pd.DataFrame(recommendations)

# ============================================================
# DISPLAY
# ============================================================

print("=" * 80)
print("ENTERPRISE AI RISK — CONTROL RECOMMENDATION ENGINE")
print("=" * 80)

print(f"\nAI System : {system['use_case_name']}")
print(f"Industry  : {system['industry']}")

print("\n" + "-" * 80)

print("IDENTIFIED CONTROL GAPS")

for gap in gaps:

    print(f"• {gap}")

print("\n" + "-" * 80)

print("RECOMMENDED CONTROL PLAN")

for i, row in control_plan.iterrows():

    print(f"\n{i + 1}. {row['control']}")

    print(
        f"   Risk Driver : {row['risk_driver']}"
    )

    print(
        f"   Owner       : {row['owner']}"
    )

    print(
        f"   Priority    : {row['priority']}"
    )

    print(
        f"   Action      : {row['recommended_action']}"
    )

# ============================================================
# SAVE FOR POWER BI / GENAI
# ============================================================

control_plan.to_csv(
    "data/control_recommendations.csv",
    index=False
)

print("\n" + "-" * 80)

print("Control plan saved to:")
print("data/control_recommendations.csv")

print("\n" + "=" * 80)
print("CONTROL RECOMMENDATION ANALYSIS COMPLETE")
print("=" * 80)