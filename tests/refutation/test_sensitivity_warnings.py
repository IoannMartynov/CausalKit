import warnings

from causalkit.refutation.unconfoundedness.uncofoundedness_validation import sensitivity_analysis_set, sensitivity_analysis


class FakeModelWarnSB:
    def __init__(self):
        self.calls = []

    def sensitivity_benchmark(self, benchmarking_set, **kwargs):
        # Emit the sklearn FutureWarning message we want to suppress
        warnings.warn(
            "'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 and will be removed in 1.8.",
            FutureWarning,
        )
        self.calls.append(list(benchmarking_set))
        return tuple(benchmarking_set)


class FakeModelWarnSA:
    def __init__(self):
        self.called = False
        self.sensitivity_summary = None

    def sensitivity_analysis(self, cf_y, cf_d, rho):
        warnings.warn(
            "'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 and will be removed in 1.8.",
            FutureWarning,
        )
        self.called = True
        self.sensitivity_summary = "OK"


def test_sensitivity_benchmark_futurewarning_is_suppressed():
    model = FakeModelWarnSB()
    effect = {"model": model}

    bench = ["tenure_months", "avg_sessions_week", "spend_last_month"]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = sensitivity_analysis_set(effect, bench)
    # Ensure the specific sklearn FutureWarning is not present
    assert not any(
        isinstance(rec.message, FutureWarning)
        and "force_all_finite" in str(rec.message)
        for rec in w
    )

    # Ensure we still get the expected structure and calls
    assert isinstance(res, dict)
    assert set(res.keys()) == set(bench)
    assert len(model.calls) == 3


def test_sensitivity_analysis_futurewarning_is_suppressed():
    model = FakeModelWarnSA()
    effect = {"model": model}

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = sensitivity_analysis(effect, cf_y=0.01, cf_d=0.02, rho=1.0)
    assert not any(
        isinstance(rec.message, FutureWarning)
        and "force_all_finite" in str(rec.message)
        for rec in w
    )

    assert model.called is True
    assert out == "OK"
