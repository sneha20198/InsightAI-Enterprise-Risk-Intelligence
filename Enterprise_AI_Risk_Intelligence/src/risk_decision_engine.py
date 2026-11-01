import pandas as pd
import joblib
import shap

# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("models/ai_risk_model.pkl")

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("data/ai_use_cases.csv")

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

# ============================================================
# SELECT SYSTEM
# ============================================================

system_index = 0

system = df.iloc[system_index]

X_system = df[features].iloc[[system_index]]

# ============================================================
# ML PREDICTION
# ============================================================

prediction = model.predict(X_system)[0]

probabilities = model.predict_proba(X_system)[0]

classes = model.classes_

probability_map = dict(
    zip(classes, probabilities)
)

confidence = max(probabilities)

# ============================================================
# SHAP EXPLANATION
# ============================================================

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_system)

if isinstance(shap_values, list):

    class_index = list(classes).index(prediction)

    impacts = shap_values[class_index][0]

else:

    if len(shap_values.shape) == 3:

        class_index = list(classes).index(prediction)

        impacts = shap_values[0, :, class_index]

    else:

        impacts = shap_values[0]

impacts = impacts.reshape(-1)

explanation = pd.DataFrame({
    "feature": features,
    "impact": impacts
})

explanation["absolute_impact"] = explanation["impact"].abs()

explanation = explanation.sort_values(
    "absolute_impact",
    ascending=False
)

top_drivers = explanation.head(5)

# ============================================================
# GOVERNANCE DECISION ENGINE
# ============================================================

priority_score = 0

control_gaps = []

recommendations = []

# Regulatory exposure
if system["regulatory_exposure"] >= 3:

    priority_score += 3

    control_gaps.append(
        "High regulatory exposure"
    )

    recommendations.append(
        "Perform regulatory and compliance assessment."
    )

# Business criticality
if system["business_criticality"] >= 3:

    priority_score += 3

    control_gaps.append(
        "High business criticality"
    )

    recommendations.append(
        "Introduce enhanced governance and approval controls."
    )

# Human oversight
if system["human_oversight"] >= 2:

    priority_score += 2

    control_gaps.append(
        "Insufficient human oversight"
    )

    recommendations.append(
        "Strengthen human-in-the-loop review."
    )

# Data sensitivity
if system["data_sensitivity"] >= 3:

    priority_score += 2

    control_gaps.append(
        "High data sensitivity"
    )

    recommendations.append(
        "Apply stronger data protection and access controls."
    )

# Bias
if system["bias_risk"] >= 2:

    priority_score += 2

    control_gaps.append(
        "Elevated bias risk"
    )

    recommendations.append(
        "Perform fairness and bias testing."
    )

# Transparency
if system["model_transparency"] >= 2:

    priority_score += 2

    control_gaps.append(
        "Limited model transparency"
    )

    recommendations.append(
        "Improve model documentation and explainability."
    )

# Error rate
if system["error_rate"] >= 0.10:

    priority_score += 2

    control_gaps.append(
        "Elevated model error rate"
    )

    recommendations.append(
        "Introduce continuous model validation and monitoring."
    )

# ============================================================
# PRIORITY CLASSIFICATION
# ============================================================

if priority_score >= 8:

    priority = "CRITICAL"

elif priority_score >= 5:

    priority = "HIGH"

elif priority_score >= 3:

    priority = "MEDIUM"

else:

    priority = "LOW"

# ============================================================
# PRINT EXECUTIVE DECISION
# ============================================================

print("=" * 70)
print("ENTERPRISE AI GOVERNANCE DECISION")
print("=" * 70)

print(f"\nAI System      : {system['use_case_name']}")
print(f"Industry       : {system['industry']}")
print(f"Business Area  : {system['business_function']}")

print("\n--------------------------------------------------")

print(f"ML Risk        : {prediction}")
print(f"Confidence     : {confidence:.2%}")
print(f"Risk Score     : {system['risk_score']:.2f}")

print(f"\nGovernance Priority : {priority}")
print(f"Priority Score      : {priority_score}")

print("\n--------------------------------------------------")

print("\nTOP RISK DRIVERS")

for i, (_, row) in enumerate(
    top_drivers.iterrows(),
    1
):

    direction = (
        "increases risk"
        if row["impact"] > 0
        else "reduces risk"
    )

    print(
        f"{i}. {row['feature']} "
        f"({direction})"
    )

print("\n--------------------------------------------------")

print("\nCONTROL GAPS")

if control_gaps:

    for i, gap in enumerate(
        control_gaps,
        1
    ):

        print(f"{i}. {gap}")

else:

    print("No major control gaps detected.")

print("\n--------------------------------------------------")

print("\nRECOMMENDED ACTIONS")

if recommendations:

    for i, recommendation in enumerate(
        recommendations,
        1
    ):

        print(
            f"{i}. {recommendation}"
        )

else:

    print(
        "Continue standard AI governance monitoring."
    )

print("\n" + "=" * 70)
print("GOVERNANCE DECISION COMPLETE")
print("=" * 70)