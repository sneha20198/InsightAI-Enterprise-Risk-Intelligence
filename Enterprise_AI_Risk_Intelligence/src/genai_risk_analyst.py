import pandas as pd
from pathlib import Path

# ============================================================
# ENTERPRISE AI — GENAI RISK ANALYST
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# Load existing governance outputs
risk_file = BASE_DIR / "data" / "ai_use_cases.csv"
control_file = BASE_DIR / "data" / "control_recommendations.csv"

risk_data = pd.read_csv(risk_file)
control_data = pd.read_csv(control_file)

# ------------------------------------------------------------
# Select the AI system for executive analysis
# ------------------------------------------------------------

system = risk_data.iloc[0]

system_name = system["use_case_name"]
industry = system["industry"]
business = system["business_function"]

risk_level = system["risk_level"]
risk_score = system["risk_score"]

# ------------------------------------------------------------
# Identify relevant control recommendations
# ------------------------------------------------------------

controls = control_data.copy()

# ------------------------------------------------------------
# Generate executive risk brief
# ------------------------------------------------------------

print("=" * 75)
print("ENTERPRISE AI — GENAI EXECUTIVE RISK ANALYST")
print("=" * 75)

print()
print("AI SYSTEM      :", system_name)
print("INDUSTRY       :", industry)
print("BUSINESS AREA  :", business)

print()
print("-" * 75)
print("EXECUTIVE RISK ASSESSMENT")
print("-" * 75)

print(
    f"The {system_name} is currently classified as "
    f"{risk_level} risk with a risk score of {risk_score:.2f}."
)

if risk_level == "High":
    assessment = (
        "Immediate governance intervention is recommended before "
        "the AI system is allowed to operate with high autonomy."
    )
elif risk_level == "Medium":
    assessment = (
        "The system requires enhanced governance controls, "
        "continuous monitoring and appropriate human oversight."
    )
else:
    assessment = (
        "The system presents comparatively lower risk but should "
        "remain subject to periodic governance review."
    )

print(assessment)

print()
print("-" * 75)
print("GOVERNANCE RECOMMENDATIONS")
print("-" * 75)

for i, row in controls.head(4).iterrows():

    control_name = row.get("control_name", row.get("control", "Governance Control"))
    owner = row.get("owner", row.get("control_owner", "AI Governance"))
    priority = row.get("priority", row.get("control_priority", "High"))
    action = row.get("action", row.get("recommended_action", "Implement appropriate governance control."))

    print(f"{i + 1}. {control_name}")
    print(f"   Owner    : {owner}")
    print(f"   Priority : {priority}")
    print(f"   Action   : {action}")
    print()

if risk_level == "High":
    decision = "ESCALATE FOR IMMEDIATE GOVERNANCE REVIEW"
elif risk_level == "Medium":
    decision = "PROCEED WITH ENHANCED GOVERNANCE CONTROLS"
else:
    decision = "PROCEED WITH STANDARD GOVERNANCE MONITORING"

print(decision)

print()
print("=" * 75)
print("GENAI RISK ANALYSIS COMPLETE")
print("=" * 75)