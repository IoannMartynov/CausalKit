"""
Simple test script to verify the calculate_mde function.
"""

from causalkit.design import calculate_mde

# Test for conversion data
print("Testing MDE calculation for conversion data:")
result_conv = calculate_mde(
    sample_size=1000,
    baseline_rate=0.1,
    data_type='conversion'
)
print(f"MDE (absolute): {result_conv['mde']:.4f}")
print(f"MDE (relative): {result_conv['mde_relative']:.4f} or {result_conv['mde_relative']*100:.2f}%")
print("Parameters used:", result_conv['parameters'])
print()

# Test for continuous data
print("Testing MDE calculation for continuous data:")
result_cont = calculate_mde(
    sample_size=(500, 500),
    variance=4,
    baseline_rate=10,  # Optional for continuous data
    data_type='continuous'
)
print(f"MDE (absolute): {result_cont['mde']:.4f}")
print(f"MDE (relative): {result_cont['mde_relative']:.4f} or {result_cont['mde_relative']*100:.2f}%")
print("Parameters used:", result_cont['parameters'])
print()

# Test with different sample allocation
print("Testing MDE calculation with different sample allocation:")
result_ratio = calculate_mde(
    sample_size=1000,
    baseline_rate=0.1,
    data_type='conversion',
    ratio=0.7  # 70% in control, 30% in treatment
)
print(f"MDE (absolute): {result_ratio['mde']:.4f}")
print(f"MDE (relative): {result_ratio['mde_relative']:.4f} or {result_ratio['mde_relative']*100:.2f}%")
print("Parameters used:", result_ratio['parameters'])
print()

# Test with different alpha and power
print("Testing MDE calculation with different alpha and power:")
result_power = calculate_mde(
    sample_size=1000,
    baseline_rate=0.1,
    data_type='conversion',
    alpha=0.01,  # More stringent significance level
    power=0.9    # Higher power
)
print(f"MDE (absolute): {result_power['mde']:.4f}")
print(f"MDE (relative): {result_power['mde_relative']:.4f} or {result_power['mde_relative']*100:.2f}%")
print("Parameters used:", result_power['parameters'])