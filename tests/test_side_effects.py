"""Tests for the side-effect lexicon."""
from src.analysis.side_effects import detect_side_effects


def test_detects_multiple_effects():
    found = detect_side_effects("Terrible nausea, constant headaches and I gained 10 lbs.")
    assert {"nausea", "headache", "weight gain"} <= set(found)


def test_whole_words_only():
    assert "fatigue" not in detect_side_effects("I recently retired and feel great")


def test_condition_is_not_counted_as_side_effect():
    text = "My depression is much better but I feel dizzy"
    assert "depression" in detect_side_effects(text)
    found = detect_side_effects(text, condition="Depression")
    assert "depression" not in found and "dizziness" in found


def test_empty_input():
    assert detect_side_effects(None) == []


def test_mental_health_conditions_overlap():
    found = detect_side_effects("Less depressed but anxiety got worse and nausea", condition="Bipolar Disorde")
    assert "depression" not in found and "anxiety" not in found and "nausea" in found
