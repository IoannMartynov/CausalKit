import numpy as np
import pandas as pd
import doubleml as dml
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Create a simple dataset
np.random.seed(42)
n = 1000

# Generate covariates
age = np.random.normal(50, 10, n)
smoker = np.random.binomial(1, 0.3, n)
bmi = np.random.normal(27, 4, n)

# Generate treatment (with confounding)
propensity = 1 / (1 + np.exp(-(0.2 * age + 0.4 * smoker - 0.3 * bmi - 0.5)))
treatment = np.random.binomial(1, propensity, n)

# Generate outcome (with heterogeneous treatment effect)
# Treatment effect varies with age
treatment_effect = 2.0 + 0.05 * (age - 50)
outcome = 1.0 * age - 0.5 * smoker + 0.2 * bmi + treatment_effect * treatment + np.random.normal(0, 1, n)

# Create DataFrame
df = pd.DataFrame({
    'age': age,
    'smoker': smoker,
    'bmi': bmi,
    'treatment': treatment,
    'outcome': outcome
})

# Create DoubleMLData object
data_dml = dml.DoubleMLData(
    df,
    y_col='outcome',
    d_cols='treatment',
    x_cols=['age', 'smoker', 'bmi']
)

# Set up ML models
ml_g = RandomForestRegressor(n_estimators=100, max_features=3, max_depth=5, min_samples_leaf=2, random_state=42)
ml_m = RandomForestClassifier(n_estimators=100, max_features=3, max_depth=5, min_samples_leaf=2, random_state=42)

# Create and fit DoubleMLIRM object
dml_irm_obj = dml.DoubleMLIRM(
    data_dml,
    ml_g=ml_g,
    ml_m=ml_m,
    n_folds=5,
    n_rep=1,
    score="ATE"
)

dml_irm_obj.fit()

# Print available attributes and methods
print("Available attributes and methods:")
for attr in dir(dml_irm_obj):
    if not attr.startswith('_'):
        print(f"- {attr}")
    elif attr == '_orthogonal_signals':
        print(f"- {attr} (private)")

# Check if _orthogonal_signals exists
if hasattr(dml_irm_obj, '_orthogonal_signals'):
    print("\nOrthogonal signals shape:", dml_irm_obj._orthogonal_signals.shape)
    print("First few values:", dml_irm_obj._orthogonal_signals[:5])

# Check if psi_elements exists (alternative way to access orthogonal signals)
if hasattr(dml_irm_obj, 'psi_elements'):
    print("\nPsi elements keys:", dml_irm_obj.psi_elements.keys())
    if 'psi_b' in dml_irm_obj.psi_elements:
        print("psi_b shape:", dml_irm_obj.psi_elements['psi_b'].shape)
        print("First few values:", dml_irm_obj.psi_elements['psi_b'][:5])

# Check CATE estimation methods
print("\nChecking CATE methods:")

# Check if cate method exists
if hasattr(dml_irm_obj, 'cate'):
    print("cate method exists")
    # Create a basis for CATE estimation
    basis = pd.DataFrame({'age': age})
    try:
        cate_model = dml_irm_obj.cate(basis)
        print("CATE model created successfully")
        print("CATE model type:", type(cate_model))
    except Exception as e:
        print(f"Error creating CATE model: {e}")

# Check if blp_predict method exists
if hasattr(dml_irm_obj, 'blp_predict'):
    print("\nblp_predict method exists")
    # Create new data for prediction
    X_new = pd.DataFrame({
        'age': [40, 50, 60],
        'smoker': [0, 1, 0],
        'bmi': [25, 30, 28]
    })
    try:
        predictions = dml_irm_obj.blp_predict(X_new)
        print("Predictions made successfully")
        print("Predictions:", predictions)
    except Exception as e:
        print(f"Error making predictions: {e}")