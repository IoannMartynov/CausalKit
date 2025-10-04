"""
Test for the DoubleML implementation in causalkit.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from causalkit.data import CausalData
from causalkit.inference.ate import dml_ate_source


def test_dml():
    """
    Test the dml function with a simple example.
    """
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
    
    # Generate outcome (with treatment effect)
    true_effect = 2.0
    outcome = 1.0 * age - 0.5 * smoker + 0.2 * bmi + true_effect * treatment + np.random.normal(0, 1, n)
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'smoker': smoker,
        'bmi': bmi,
        'treatment': treatment,
        'outcome': outcome
    })
    
    # Create CausalData object
    causal_data = CausalData(
        df=df,
        treatment='treatment',
        outcome='outcome',
        confounders=['age', 'smoker', 'bmi']
    )
    
    # Set up ML models
    ml_g = RandomForestRegressor(n_estimators=100, max_features=3, max_depth=5, min_samples_leaf=2, random_state=42)
    ml_m = RandomForestClassifier(n_estimators=100, max_features=3, max_depth=5, min_samples_leaf=2, random_state=42)
    
    # Run DML
    results = dml_ate_source(
        data=causal_data,
        ml_g=ml_g,
        ml_m=ml_m,
        n_folds=5,
        n_rep=1,
        score="ATE",
        confidence_level=0.95
    )
    
    # Print results
    print(f"True effect: {true_effect}")
    print(f"Estimated effect: {results['coefficient']:.4f}")
    print(f"Standard error: {results['std_error']:.4f}")
    print(f"P-value: {results['p_value']:.4f}")
    print(f"95% CI: {results['confidence_interval']}")
    
    # Check that the true effect is within the confidence interval
    assert results['confidence_interval'][0] <= true_effect <= results['confidence_interval'][1], \
        f"True effect {true_effect} not in confidence interval {results['confidence_interval']}"
    
    # Check that the estimated effect is close to the true effect
    assert abs(results['coefficient'] - true_effect) < 0.5, \
        f"Estimated effect {results['coefficient']} too far from true effect {true_effect}"
    
    print("All tests passed!")


if __name__ == "__main__":
    test_dml()