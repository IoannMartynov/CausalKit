import pytest

from causalkit.data.generators import generate_rct, CausalDatasetGenerator


def test_deterministic_user_ids_reproducible():
    df1 = generate_rct(n=100, random_state=123, add_ancillary=True, deterministic_ids=True)
    df2 = generate_rct(n=100, random_state=123, add_ancillary=True, deterministic_ids=True)
    assert list(df1["user_id"]) == list(df2["user_id"])  # identical across runs with same seed


def test_nondeterministic_user_ids_change_with_same_seed():
    df1 = generate_rct(n=50, random_state=123, add_ancillary=True, deterministic_ids=False)
    df2 = generate_rct(n=50, random_state=123, add_ancillary=True, deterministic_ids=False)
    assert list(df1["user_id"]) != list(df2["user_id"])  # should differ due to uuid4()


def test_to_causal_data_accepts_single_string_and_preserves_order():
    gen = CausalDatasetGenerator(k=3, seed=0)
    # Single string accepted
    cd = gen.to_causal_data(100, confounders="x1")
    assert hasattr(cd, "confounders")
    assert cd.confounders == ["x1"]

    # When None, keep order of columns in DataFrame excluding known non-confounders
    cd2 = gen.to_causal_data(30)
    exclude = {"y", "t", "m", "g0", "g1", "cate", "propensity", "mu0", "mu1"}
    expected = [c for c in cd2.df.columns if c not in exclude]
    assert cd2.confounders == expected


def test_generator_uses_slots():
    gen = CausalDatasetGenerator(seed=0)
    with pytest.raises(AttributeError):
        gen.some_new_attr = 123
