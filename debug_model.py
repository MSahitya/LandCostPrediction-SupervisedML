import joblib

model = joblib.load("land_cost_predictor_pipeline.joblib")
print("Type:", type(model))

# Print feature names if available
if hasattr(model, "feature_names_in_"):
    print("Features:", list(model.feature_names_in_))
elif hasattr(model, "steps"):
    first_step = model.steps[0][1]
    if hasattr(first_step, "feature_names_in_"):
        print("Features:", list(first_step.feature_names_in_))
    elif hasattr(first_step, "get_feature_names_out"):
        print("Features out:", list(first_step.get_feature_names_out()))
    else:
        print("First step:", first_step)
        print("First step params:", first_step.get_params())

