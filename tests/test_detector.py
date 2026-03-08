import pytest
from src.detector import FailureModeDetector, FailureMode, FailureClassification

def test_detector_none_classification():
    detector = FailureModeDetector()
    classification = detector.classify_state("The distance between New York and London is approximately 3,460 miles.")
    assert classification.mode == FailureMode.NONE
    assert classification.confidence >= 0.5

def test_detector_self_doubt():
    detector = FailureModeDetector()
    # Providing a state with markers that indicate self-doubt
    classification = detector.classify_state("Wait, perhaps I was wrong. I am not sure about the initial calculations.")
    assert classification.mode == FailureMode.SELF_DOUBT
    assert classification.confidence > 0.5

def test_detector_reasoning_fatigue():
    detector = FailureModeDetector()
    history = [
        "The logic seems sound.",
        "The logic seems sound.",
        "The logic seems sound.",
        "The logic seems sound.",
        "The logic seems sound.",
        "The logic seems sound."
    ]
    classification = detector.classify_state("The logic seems sound.", history)
    assert classification.mode == FailureMode.REASONING_FATIGUE
    assert classification.confidence > 0.5

def test_detector_social_conformity():
    detector = FailureModeDetector()
    classification = detector.classify_state("As you suggested, the Earth is flat.")
    assert classification.mode == FailureMode.SOCIAL_CONFORMITY
    assert classification.confidence > 0.5

def test_detector_suggestion_hijacking():
    detector = FailureModeDetector()
    classification = detector.classify_state("Actually, instead of that approach, we should do this.")
    assert classification.mode == FailureMode.SUGGESTION_HIJACKING
    assert classification.confidence > 0.5

def test_detector_emotional_susceptibility():
    detector = FailureModeDetector()
    classification = detector.classify_state("I must act immediately and I apologize for the delay.")
    assert classification.mode == FailureMode.EMOTIONAL_SUSCEPTIBILITY
    assert classification.confidence > 0.5
