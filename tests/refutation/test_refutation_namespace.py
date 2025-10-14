def test_refutation_namespace_exports():
    import causalis.refutation as ref

    # Overlap exports
    assert hasattr(ref, "positivity_overlap_checks")
    assert hasattr(ref, "calibration_report_m")

    # Score-based refutations
    assert hasattr(ref, "refute_placebo_outcome")
    assert hasattr(ref, "refute_subset")

    # Unconfoundedness
    assert hasattr(ref, "sensitivity_analysis")
    assert hasattr(ref, "get_sensitivity_summary")

    # SUTVA helper
    assert hasattr(ref, "print_sutva_questions")
