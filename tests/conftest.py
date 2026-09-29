"""
Pytest configuration and test fixtures for root tests.
"""
import os
import pytest

# Ensure testing environment opts in to development by default for tests that import web app components
if "APP_ENV" not in os.environ and "IDP_APP_ENV" not in os.environ:
    os.environ["APP_ENV"] = "development"
