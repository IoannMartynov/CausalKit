import sys
from io import StringIO

from causalis.refutation.sutva.sutva_validation import (
    print_sutva_questions,
    QUESTIONS,
)


def test_print_sutva_questions_prints_expected_lines(capsys):
    # Call the function
    print_sutva_questions()

    # Capture the output
    captured = capsys.readouterr()
    out_lines = captured.out.strip().splitlines()

    # Expected lines must match exactly
    expected_lines = list(QUESTIONS)
    assert out_lines == expected_lines


def test_import_has_no_side_effects():
    # Ensure importing the module does not print anything by reloading it in isolation
    import importlib

    module_name = "causalis.refutation.sutva.sutva_validation"

    # Remove from sys.modules to force a clean import
    if module_name in sys.modules:
        del sys.modules[module_name]

    # Capture stdout during fresh import
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        importlib.import_module(module_name)
        output = sys.stdout.getvalue()
    finally:
        sys.stdout = old_stdout

    assert output == ""
