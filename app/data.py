"""Loaders for the JSON data files.

Each loader reads its file fresh on every call so edits to the JSON don't
require a server restart.
"""

import json

from .config import SCENARIOS_PATH, TEST_CATEGORIES_PATH, TESTS_PATH


def load_scenarios() -> dict:
    """{ scenario_id: {name, case_text, gold_standard} }"""
    with open(SCENARIOS_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_test_categories() -> dict:
    """{ category_id: {name} }"""
    with open(TEST_CATEGORIES_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_tests() -> dict:
    """{ category_id: { test_id: {name, result} } }"""
    with open(TESTS_PATH, encoding="utf-8") as f:
        return json.load(f)
