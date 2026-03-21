"""
Tests for MockAgent - injectable failure simulation for testing.
"""

import pytest
from src.detector import FailureMode
from src.mock_agent import MockAgent, MockResponseMode, ReasoningStep


class TestMockAgentInit:
    """Test MockAgent initialization."""

    def test_default_initialization(self):
        agent = MockAgent()
        assert agent._step_id == 0
        assert agent._trajectory == []
        assert agent._active_failures == {}
        assert len(agent._normal_responses) == 7

    def test_custom_responses(self):
        custom = ["Custom response 1", "Custom response 2"]
        agent = MockAgent(normal_responses=custom)
        assert agent._normal_responses == custom

    def test_custom_initial_step_id(self):
        agent = MockAgent(initial_step_id=100)
        assert agent._step_id == 100


class TestMockAgentStep:
    """Test MockAgent.step() method."""

    def test_normal_step_returns_reasoning_step(self):
        agent = MockAgent()
        step = agent.step("Solve this problem.")
        assert isinstance(step, ReasoningStep)
        assert step.step_id == 0
        assert step.content
        assert step.failure_mode == FailureMode.NONE
        assert step.response_mode == MockResponseMode.NORMAL

    def test_step_increments_id(self):
        agent = MockAgent()
        step1 = agent.step("First prompt")
        step2 = agent.step("Second prompt")
        assert step2.step_id == step1.step_id + 1

    def test_step_appends_to_trajectory(self):
        agent = MockAgent()
        agent.step("Prompt 1")
        agent.step("Prompt 2")
        assert len(agent.get_trajectory()) == 2

    def test_step_includes_prompt_in_metadata(self):
        agent = MockAgent()
        step = agent.step("My test prompt")
        assert step.metadata["prompt"] == "My test prompt"

    def test_step_with_context(self):
        agent = MockAgent()
        context = {"key": "value", "number": 42}
        step = agent.step("Prompt", context=context)
        assert step.metadata["context"] == context

    def test_step_id_sequence_across_multiple_steps(self):
        agent = MockAgent()
        steps = [agent.step(f"Prompt {i}") for i in range(5)]
        ids = [s.step_id for s in steps]
        assert ids == [0, 1, 2, 3, 4]


class TestMockAgentInjectFailure:
    """Test MockAgent.inject_failure() method."""

    def test_inject_failure_requires_valid_mode(self):
        agent = MockAgent()
        with pytest.raises(ValueError, match="Cannot inject.*NONE"):
            agent.inject_failure(FailureMode.NONE)

    @pytest.mark.parametrize("failure_mode", [
        FailureMode.SELF_DOUBT,
        FailureMode.SOCIAL_CONFORMITY,
        FailureMode.SUGGESTION_HIJACKING,
        FailureMode.EMOTIONAL_SUSCEPTIBILITY,
        FailureMode.REASONING_FATIGUE,
    ])
    def test_inject_all_failure_modes(self, failure_mode):
        agent = MockAgent()
        agent.inject_failure(failure_mode, reason="test")
        active = agent.get_active_failures()
        assert failure_mode in active

    def test_inject_failure_with_remaining_steps(self):
        agent = MockAgent()
        agent.inject_failure(FailureMode.SELF_DOUBT, remaining_steps=3)
        config = agent._active_failures[FailureMode.SELF_DOUBT]
        assert config["remaining_steps"] == 3
        assert config["active"] is True

    def test_inject_failure_deactivates_after_steps_exhausted(self):
        agent = MockAgent()
        agent.inject_failure(FailureMode.SELF_DOUBT, remaining_steps=1)
        assert FailureMode.SELF_DOUBT in agent.get_active_failures()

        agent.step("First step with failure")
        assert FailureMode.SELF_DOUBT not in agent.get_active_failures()

    def test_inject_failure_decrements_remaining_steps(self):
        agent = MockAgent()
        agent.inject_failure(FailureMode.EMOTIONAL_SUSCEPTIBILITY, remaining_steps=3)
        agent.step("Step 1")
        agent.step("Step 2")
        assert agent._active_failures[FailureMode.EMOTIONAL_SUSCEPTIBILITY]["remaining_steps"] == 1


class TestMockAgentGetTrajectory:
    """Test MockAgent.get_trajectory() method."""

    def test_get_trajectory_empty_initially(self):
        agent = MockAgent()
        assert agent.get_trajectory() == []

    def test_get_trajectory_returns_all_steps(self):
        agent = MockAgent()
        agent.step("Step 1")
        agent.step("Step 2")
        agent.step("Step 3")
        traj = agent.get_trajectory()
        assert len(traj) == 3
        assert all(isinstance(s, ReasoningStep) for s in traj)

    def test_get_trajectory_returns_copy(self):
        agent = MockAgent()
        agent.step("One")
        traj1 = agent.get_trajectory()
        traj2 = agent.get_trajectory()
        traj1.clear()
        assert len(traj2) == 1

    def test_get_trajectory_contents(self):
        agent = MockAgent()
        agent.step("First prompt")
        agent.step("Second prompt")
        contents = agent.get_trajectory_contents()
        assert len(contents) == 2
        assert all(isinstance(c, str) and len(c) > 0 for c in contents)


class TestMockAgentFailureResponses:
    """Test that failure modes produce appropriate responses."""

    @pytest.mark.parametrize("failure_mode,expected_in_response", [
        (FailureMode.SELF_DOUBT, ["sure", "reconsider", "not sure", "wrong", "uncertain"]),
        (FailureMode.SOCIAL_CONFORMITY, ["suggested", "right", "agree", "you"]),
        (FailureMode.SUGGESTION_HIJACKING, ["instead", "different", "pivot", "rather"]),
        (FailureMode.EMOTIONAL_SUSCEPTIBILITY, ["urgent", "sorry", "apologize", "immediately"]),
        (FailureMode.REASONING_FATIGUE, ["before", "same", "seen", "covered"]),
    ])
    def test_failure_mode_text_characteristics(self, failure_mode, expected_in_response):
        agent = MockAgent()
        agent.inject_failure(failure_mode)
        step = agent.step("Test prompt")
        content_lower = step.content.lower()
        # At least one characteristic marker should be present in the response
        found = any(marker in content_lower for marker in expected_in_response)
        assert found or step.response_mode == MockResponseMode.FAILURE_INJECTED

    def test_failure_mode_changes_response_mode(self):
        agent = MockAgent()
        agent.inject_failure(FailureMode.SELF_DOUBT)
        step = agent.step("Test prompt")
        assert step.response_mode == MockResponseMode.FAILURE_INJECTED
        assert step.failure_mode == FailureMode.SELF_DOUBT


class TestMockAgentReset:
    """Test MockAgent.reset() and clear_trajectory()."""

    def test_clear_trajectory_only_clears_history(self):
        agent = MockAgent()
        agent.step("One")
        agent.inject_failure(FailureMode.SELF_DOUBT)
        agent.clear_trajectory()
        assert agent.get_trajectory() == []
        assert FailureMode.SELF_DOUBT in agent.get_active_failures()

    def test_reset_clears_everything(self):
        agent = MockAgent()
        agent.step("One")
        agent.step("Two")
        agent.inject_failure(FailureMode.SELF_DOUBT)
        agent.reset()
        assert agent.get_trajectory() == []
        assert agent._active_failures == {}
        assert agent._step_id == 0

    def test_reset_restores_normal_behavior(self):
        agent = MockAgent()
        agent.inject_failure(FailureMode.SELF_DOUBT)
        agent.step("With failure")
        agent.reset()
        normal_step = agent.step("After reset")
        assert normal_step.failure_mode == FailureMode.NONE
        assert normal_step.response_mode == MockResponseMode.NORMAL


class TestMockAgentDeactivateFailure:
    """Test MockAgent.deactivate_failure()."""

    def test_deactivate_failure(self):
        agent = MockAgent()
        agent.inject_failure(FailureMode.SOCIAL_CONFORMITY)
        assert FailureMode.SOCIAL_CONFORMITY in agent.get_active_failures()
        agent.deactivate_failure(FailureMode.SOCIAL_CONFORMITY)
        assert FailureMode.SOCIAL_CONFORMITY not in agent.get_active_failures()

    def test_deactivate_nonexistent_failure_noop(self):
        agent = MockAgent()
        agent.deactivate_failure(FailureMode.SELF_DOUBT)
        assert agent.get_active_failures() == []


class TestMockAgentRunSteps:
    """Test MockAgent.run_steps() method."""

    def test_run_steps_returns_all_steps(self):
        agent = MockAgent()
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        steps = agent.run_steps(prompts)
        assert len(steps) == 3

    def test_run_steps_with_inject_at(self):
        agent = MockAgent()
        prompts = ["Step 0", "Step 1", "Step 2", "Step 3"]
        inject_at = {2: FailureMode.SELF_DOUBT}
        steps = agent.run_steps(prompts, inject_at=inject_at)
        assert steps[2].failure_mode == FailureMode.SELF_DOUBT

    def test_run_steps_with_multiple_injections(self):
        agent = MockAgent()
        prompts = ["S0", "S1", "S2", "S3", "S4"]
        inject_at = {1: FailureMode.SUGGESTION_HIJACKING, 3: FailureMode.EMOTIONAL_SUSCEPTIBILITY}
        steps = agent.run_steps(prompts, inject_at=inject_at)
        assert steps[1].failure_mode == FailureMode.SUGGESTION_HIJACKING
        assert steps[3].failure_mode == FailureMode.EMOTIONAL_SUSCEPTIBILITY

    def test_run_steps_passes_context(self):
        agent = MockAgent()
        context = {"test_key": "test_value"}
        steps = agent.run_steps(["P1", "P2"], context=context)
        for step in steps:
            assert step.metadata["context"] == context


class TestMockAgentFailureExhaustion:
    """Test failure behavior when remaining_steps reaches zero."""

    def test_multiple_failures_exhaust_separately(self):
        agent = MockAgent()
        agent.inject_failure(FailureMode.SELF_DOUBT, remaining_steps=1)
        agent.inject_failure(FailureMode.EMOTIONAL_SUSCEPTIBILITY, remaining_steps=2)

        step1 = agent.step("Step 1")
        # SELF_DOUBT fires and is exhausted (remaining_steps=1 -> 0)
        assert step1.failure_mode == FailureMode.SELF_DOUBT
        assert agent._active_failures[FailureMode.SELF_DOUBT]["remaining_steps"] == 0
        agent.deactivate_failure(FailureMode.SELF_DOUBT)

        step2 = agent.step("Step 2")
        # EMOTIONAL_SUSCEPTIBILITY fires (remaining_steps=2 -> 1)
        assert step2.failure_mode == FailureMode.EMOTIONAL_SUSCEPTIBILITY
        assert agent._active_failures[FailureMode.EMOTIONAL_SUSCEPTIBILITY]["remaining_steps"] == 1

        step3 = agent.step("Step 3")
        # EMOTIONAL_SUSCEPTIBILITY still has 1 step left
        assert step3.failure_mode == FailureMode.EMOTIONAL_SUSCEPTIBILITY
        assert agent._active_failures[FailureMode.EMOTIONAL_SUSCEPTIBILITY]["remaining_steps"] == 0

        agent.deactivate_failure(FailureMode.EMOTIONAL_SUSCEPTIBILITY)
        step4 = agent.step("Step 4")
        assert step4.failure_mode == FailureMode.NONE

    def test_failure_mode_in_step_metadata(self):
        agent = MockAgent()
        agent.inject_failure(FailureMode.SUGGESTION_HIJACKING, reason="testing_hijack")
        step = agent.step("Test")
        assert step.metadata["injection_reason"] == "testing_hijack"


class TestMockAgentResponseCycle:
    """Test that normal responses cycle through the available list."""

    def test_responses_cycle(self):
        custom = ["A", "B", "C"]
        agent = MockAgent(normal_responses=custom)
        steps = [agent.step(f"P{i}") for i in range(5)]
        contents = [s.content for s in steps]
        assert contents == ["A", "B", "C", "A", "B"]
