import pytest

from causalkit.refutation.sensitivity import sensitivity_analysis_set


class FakeModelSimple:
    def __init__(self):
        self.calls = []

    # Signature with level and null_hypothesis
    def sensitivity_benchmark(self, benchmarking_set, level=0.95, null_hypothesis=0.0, **kwargs):
        # record call
        self.calls.append({
            "benchmarking_set": list(benchmarking_set),
            "level": level,
            "null_hypothesis": null_hypothesis,
            "kwargs": kwargs,
        })
        # return something identifiable for assertions
        return tuple(benchmarking_set)


class FakeModelAlpha:
    def __init__(self):
        self.calls = []

    # Signature using alpha instead of level
    def sensitivity_benchmark(self, benchmarking_set, alpha=0.05, null_hypothesis=0.0):
        self.calls.append({
            "benchmarking_set": list(benchmarking_set),
            "alpha": alpha,
            "null_hypothesis": null_hypothesis,
        })
        return {"bench": tuple(benchmarking_set), "alpha": alpha}


def make_effect(model):
    return {"model": model}


def test_list_of_strings_returns_dict_and_calls_each():
    model = FakeModelSimple()
    effect = make_effect(model)

    bench = ["tenure_months", "avg_sessions_week", "spend_last_month"]
    res = sensitivity_analysis_set(effect, bench, level=0.9, null_hypothesis=0.0)

    # Expect a dict with keys per confounder
    assert isinstance(res, dict)
    assert set(res.keys()) == set(bench)

    # Each value should be the tuple of the single benchmarking_set element
    for k in bench:
        assert res[k] == (k,)

    # Ensure there were three separate calls
    assert len(model.calls) == 3
    called_sets = [c["benchmarking_set"] for c in model.calls]
    assert called_sets == [[b] for b in bench]
    # Level propagated
    for c in model.calls:
        assert pytest.approx(c["level"], 1e-9) == 0.9
        assert c["null_hypothesis"] == 0.0


def test_grouped_lists_returns_dict_with_tuple_keys():
    model = FakeModelSimple()
    effect = make_effect(model)

    bench = [["a"], ["b", "c"]]
    res = sensitivity_analysis_set(effect, bench, level=0.95, null_hypothesis=1.23)

    assert isinstance(res, dict)
    # Keys should be 'a' for singletons and ('b','c') for group
    assert set(res.keys()) == {"a", ("b", "c")}
    assert res["a"] == ("a",)
    assert res[("b", "c")] == ("b", "c")

    # Two calls
    assert len(model.calls) == 2
    assert model.calls[0]["benchmarking_set"] == ["a"]
    assert model.calls[1]["benchmarking_set"] == ["b", "c"]
    for c in model.calls:
        assert pytest.approx(c["level"], 1e-9) == 0.95
        assert c["null_hypothesis"] == 1.23


def test_single_string_returns_single_object():
    model = FakeModelSimple()
    effect = make_effect(model)

    res = sensitivity_analysis_set(effect, "only_one", level=0.8)

    # Should not be a dict, should be direct result from FakeModel
    assert res == ("only_one",)
    assert len(model.calls) == 1
    assert model.calls[0]["benchmarking_set"] == ["only_one"]


def test_invalid_level_raises():
    model = FakeModelSimple()
    effect = make_effect(model)

    with pytest.raises(ValueError):
        sensitivity_analysis_set(effect, "x", level=1.0)
    with pytest.raises(ValueError):
        sensitivity_analysis_set(effect, ["x"], level=0.0)


def test_alpha_signature_compatibility():
    model = FakeModelAlpha()
    effect = make_effect(model)

    # level=0.9 should translate to alpha=0.1
    res = sensitivity_analysis_set(effect, ["x", "y"], level=0.9)

    assert isinstance(res, dict)
    assert set(res.keys()) == {"x", "y"}
    # Called twice
    assert len(model.calls) == 2
    for c in model.calls:
        assert pytest.approx(c["alpha"], 1e-9) == 0.1
        assert c["null_hypothesis"] == 0.0
