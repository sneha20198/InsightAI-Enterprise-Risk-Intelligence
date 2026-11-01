import pandas as pd
import joblib

# ============================================================
# LOAD MODEL + DATA
# ============================================================

model = joblib.load("models/ai_risk_model.pkl")

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

current = system[features].copy()


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_risk(values):

    input_df = pd.DataFrame(
        [values],
        columns=features
    )

    prediction = model.predict(input_df)[0]

    probabilities = model.predict_proba(input_df)[0]

    classes = model.classes_

    probability_map = dict(
        zip(classes, probabilities)
    )

    return prediction, probability_map


# ============================================================
# BASELINE
# ============================================================

base_prediction, base_probabilities = predict_risk(
    current.values
)

base_high = base_probabilities.get("High", 0)
base_medium = base_probabilities.get("Medium", 0)


# ============================================================
# GOVERNANCE INTERVENTIONS
# ============================================================

scenarios = {

    "Improve Human Oversight": {
        "human_oversight": 1
    },

    "Reduce Regulatory Exposure": {
        "regulatory_exposure": 1
    },

    "Reduce Data Sensitivity": {
        "data_sensitivity": 1
    },

    "Improve Model Transparency": {
        "model_transparency": 1
    },

    "Reduce Bias Risk": {
        "bias_risk": 1
    },

    "Reduce Business Criticality": {
        "business_criticality": 1
    },

    "Reduce Automation Risk": {
        "automation_level": 1
    }
}


# ============================================================
# RUN ALL SCENARIOS
# ============================================================

results = []

for scenario_name, changes in scenarios.items():

    scenario = current.copy()

    for feature, new_value in changes.items():

        scenario[feature] = new_value

    prediction, probabilities = predict_risk(
        scenario.values
    )

    high_probability = probabilities.get(
        "High", 0
    )

    medium_probability = probabilities.get(
        "Medium", 0
    )

    high_reduction = (
        base_high - high_probability
    )

    medium_reduction = (
        base_medium - medium_probability
    )

    results.append({

        "intervention": scenario_name,

        "risk_level": prediction,

        "high_probability": high_probability,

        "medium_probability": medium_probability,

        "high_risk_reduction": high_reduction,

        "medium_risk_reduction": medium_reduction
    })


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "high_risk_reduction",
    ascending=False
)


# ============================================================
# REPORT
# ============================================================

print("=" * 78)
print("ENTERPRISE AI RISK — GOVERNANCE INTERVENTION SIMULATOR")
print("=" * 78)

print(f"\nAI System : {system['use_case_name']}")
print(f"Industry  : {system['industry']}")
print(f"Business  : {system['business_function']}")

print("\n" + "-" * 78)

print("BASELINE")

print(f"Risk Level       : {base_prediction}")
print(f"High Risk        : {base_high:.2%}")
print(f"Medium Risk      : {base_medium:.2%}")

print("\n" + "-" * 78)

print("INTERVENTION ANALYSIS")

print(
    f"{'INTERVENTION':<32}"
    f"{'RISK':<10}"
    f"{'HIGH':<12}"
    f"{'HIGH REDUCTION':<15}"
)

print("-" * 78)

for _, row in results_df.iterrows():

    print(
        f"{row['intervention']:<32}"
        f"{row['risk_level']:<10}"
        f"{row['high_probability']:<12.2%}"
        f"{row['high_risk_reduction']:<15.2%}"
    )


# ============================================================
# BEST INTERVENTION
# ============================================================

best = results_df.iloc[0]

print("\n" + "-" * 78)

print("RECOMMENDED INTERVENTION")

print(
    f"1. {best['intervention']}"
)

print(
    f"   High-risk probability reduction: "
    f"{best['high_risk_reduction']:.2%}"
)

print(
    f"   Resulting risk classification: "
    f"{best['risk_level']}"
)


# ============================================================
# MANAGEMENT INSIGHT
# ============================================================

print("\n" + "-" * 78)

print("MANAGEMENT INSIGHT")

if best["risk_level"] != base_prediction:

    print(
        f"The recommended intervention changes "
        f"the risk classification from "
        f"{base_prediction} → {best['risk_level']}."
    )

else:

    print(
        f"The recommended intervention does not "
        f"change the overall classification from "
        f"{base_prediction}, but materially reduces "
        f"the probability of High risk."
    )

print("\n" + "=" * 78)
print("INTERVENTION ANALYSIS COMPLETE")
print("=" * 78)