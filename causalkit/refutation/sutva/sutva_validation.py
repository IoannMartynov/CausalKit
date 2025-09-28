"""
SUTVA validation helper.

This module provides a simple function to print four SUTVA-related
questions for the user to consider. It has no side effects on import.
"""
from typing import Iterable


QUESTIONS: Iterable[str] = (
    "1.) Are your clients independent (i)?",
    "2.) Do you measure confounders, treatment, and outcome in the same intervals?",
    "3.) Do you measure confounders before treatment and outcome after?",
    "4.) Do you have a consistent label of treatment, such as if a person does not receive a treatment, he has a label 0?",
)


def print_sutva_questions() -> None:
    """Print the SUTVA validation questions.

    Just prints questions, nothing more.
    """
    for q in QUESTIONS:
        print(q)


__all__ = ["QUESTIONS", "print_sutva_questions"]
