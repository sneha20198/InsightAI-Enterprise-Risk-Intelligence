import pandas as pd
import joblib
import shap

# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load("models/ai_risk_model.pkl")

# ==========================================
# LOAD DATA
# ==========================================

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

X = df[features]

# ==========================================
# SELECT ONE AI SYSTEM
# ==========================================

system = df.iloc[0]

X_system = X.iloc[[0]]

# ==========================================
# PREDICTION
# ==========================================

prediction = model.predict(X_system)[0]

probabilities = model.predict_proba(X_system)[0]

classes = model.classes_

probability_map = dict(zip(classes, probabilities))

print("=" * 65)
print("ENTERPRISE AI RISK — EXPLAINABILITY REPORT")
print("=" * 65)

print(f"\nAI System      : {system['use_case_name']}")
print(f"Industry       : {system['industry']}")
print(f"Business Area  : {system['business_function']}")

print(f"\nPredicted Risk : {prediction}")

print("\nRisk Probabilities:")

for risk_class, probability in probability_map.items():
    print(
        f"{risk_class:<10}: "
        f"{probability:.2%}"
    )

# ==========================================
# ==========================================
# SHAP EXPLANATION
# ==========================================

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_system)

# -------------------------------------------------
# Handle SHAP output for different SHAP versions
# -------------------------------------------------

if isinstance(shap_values, list):

    # Older SHAP versions
    class_index = list(classes).index(prediction)
    values = shap_values[class_index][0]

else:

    # Newer SHAP versions
    shap_array = shap_values

    # Shape can be:
    # (samples, features, classes)
    if len(shap_array.shape) == 3:

        class_index = list(classes).index(prediction)

        values = shap_array[0, :, class_index]

    # Or:
    # (samples, features)
    elif len(shap_array.shape) == 2:

        values = shap_array[0]

    else:

        values = shap_array.flatten()


# Make absolutely sure SHAP values are 1-dimensional
values = values.reshape(-1)

# ==========================================
# BUILD EXPLANATION TABLE
# ==========================================

explanation = pd.DataFrame({
    "feature": features,
    "value": X_system.iloc[0].values,
    "impact": values
})

explanation["absolute_impact"] = explanation["impact"].abs()

explanation = explanation.sort_values(
    "absolute_impact",
    ascending=False
)

print("\nTOP RISK CONTRIBUTORS")
print("-" * 65)

for _, row in explanation.head(5).iterrows():

    direction = (
        "increases risk"
        if row["impact"] > 0
        else "reduces risk"
    )

    print(
        f"{row['feature']:<25} "
        f"value={row['value']:<8} "
        f"impact={row['impact']:.4f} "
        f"→ {direction}"
    )


# ==========================================
# RECOMMENDED CONTROLS
# ==========================================

print("\nRECOMMENDED CONTROLS")
print("-" * 65)

recommendations = []

if system["regulatory_exposure"] >= 3:
    recommendations.append(
        "Perform regulatory and compliance assessment."
    )

if system["business_criticality"] >= 3:
    recommendations.append(
        "Introduce enhanced governance and approval controls."
    )

if system["human_oversight"] >= 2:
    recommendations.append(
        "Increase human-in-the-loop review."
    )

if system["data_sensitivity"] >= 3:
    recommendations.append(
        "Apply stronger data protection and access controls."
    )

if system["bias_risk"] >= 2:
    recommendations.append(
        "Perform fairness and bias testing."
    )

if system["model_transparency"] >= 2:
    recommendations.append(
        "Improve model explainability and documentation."
    )

if system["error_rate"] >= 0.10:
    recommendations.append(
        "Introduce additional model validation and monitoring."
    )

if not recommendations:
    recommendations.append(
        "Continue standard AI governance monitoring."
    )

for i, recommendation in enumerate(recommendations, 1):

    print(
        f"{i}. {recommendation}"
    )


print("\n" + "=" * 65)
print("EXPLAINABILITY ANALYSIS COMPLETE")
print("=" * 65)