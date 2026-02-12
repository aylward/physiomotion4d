"""
Utilities for experiment notebooks (e.g. when run as tests by pytest).

When experiment tests run via pytest with --run-experiments, the test runner sets
PHYSIOMOTION_RUNNING_AS_TEST=1 so notebooks can use reduced parameters for fast runs.
"""

import os


def running_as_test() -> bool:
    """
    True when the notebook is run as a test (e.g. by pytest experiment tests).

    Use this to choose fast/small parameters (fewer iterations, fewer files, etc.)
    so test runs complete in reasonable time. When False, use full parameters
    for interactive or production runs.

    Returns:
        True if PHYSIOMOTION_RUNNING_AS_TEST is set to a truthy value
        (1, true, yes, case-insensitive); False otherwise.
    """
    return os.environ.get("PHYSIOMOTION_RUNNING_AS_TEST", "").lower() in (
        "1",
        "true",
        "yes",
    )
