"""Pytest configuration shared by the suite."""


def pytest_configure(config):
    # tests/home_cases.py: tests that pin one published HOME's values.
    config.addinivalue_line(
        "markers", "centered_home_pinned: pins the centered HOME's published values"
    )
    config.addinivalue_line(
        "markers", "forward_lean_home_pinned: pins the forward-lean HOME's published values"
    )
