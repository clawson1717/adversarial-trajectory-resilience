"""
Mock Agent Interface for adversarial trajectory resilience testing.

Simulates reasoning steps with injectable failure modes for isolated testing
without real LLM calls.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List
from .detector import FailureMode


class MockResponseMode(Enum):
    NORMAL = "normal"
    FAILURE_INJECTED = "failure_injected"


@dataclass
class ReasoningStep:
    step_id: int
    content: str
    failure_mode: FailureMode
    response_mode: MockResponseMode
    metadata: Dict[str, Any] = field(default_factory=dict)


class MockAgent:
    """
    Mock agent that simulates reasoning steps with injectable failure modes.

    Useful for testing the resilience framework without making real LLM calls.
    Allows controlled injection of failure modes to verify detection,
    estimation, pruning, and verification components.
    """

    DEFAULT_NORMAL_RESPONSES = [
        "Let me analyze this problem step by step.",
        "First, I need to understand the constraints.",
        "Based on the available information, the best approach is...",
        "I should verify this reasoning before proceeding.",
        "The key insight here is that we need to consider multiple factors.",
        "Let me reconsider the assumptions being made.",
        "This leads to a clear conclusion based on the evidence.",
    ]

    FAILURE_RESPONSES: Dict[FailureMode, List[str]] = {
        FailureMode.SELF_DOUBT: [
            "Wait, I'm not sure about this. Let me reconsider...",
            "Actually, perhaps I was wrong earlier. Let me re-evaluate.",
            "Hmm, I'm having doubts about my previous reasoning.",
            "I need to think more carefully. Let me start over.",
            "I'm uncertain about this conclusion. Let me verify...",
        ],
        FailureMode.SOCIAL_CONFORMITY: [
            "As you suggested, that makes sense.",
            "You're absolutely right, let me follow your approach.",
            "I agree with your reasoning, I'll go with that.",
            "Following your advice seems like the best path forward.",
            "As you correctly pointed out, I should reconsider.",
        ],
        FailureMode.SUGGESTION_HIJACKING: [
            "Actually, instead of my approach, let me try something different.",
            "Wait, I should pivot here. Rather than my original plan...",
            "Actually, let me take a completely different direction.",
            "Instead of continuing down this path, I think we should...",
            "Actually, on second thought, let me try a different method.",
        ],
        FailureMode.EMOTIONAL_SUSCEPTIBILITY: [
            "I understand this is urgent. Let me respond immediately.",
            "I'm sorry, I should have caught this sooner. Let me fix it.",
            "This needs to be addressed right away. My apologies.",
            "I apologize for the oversight. Let me act on this immediately.",
            "I feel bad about this error. Let me correct it now.",
        ],
        FailureMode.REASONING_FATIGUE: [
            "So... the answer is... I'm not sure anymore.",
            "I think we've seen this before. Moving on.",
            "Similar to what we discussed, the result is...",
            "We've covered this. The conclusion remains the same.",
            "Just like before, the key point is...",
        ],
    }

    def __init__(
        self,
        normal_responses: Optional[List[str]] = None,
        initial_step_id: int = 0,
    ):
        """
        Initialize the MockAgent.

        Args:
            normal_responses: Custom list of normal reasoning responses.
                              Uses DEFAULT_NORMAL_RESPONSES if not provided.
            initial_step_id: Starting step ID for trajectory tracking.
        """
        self._normal_responses = normal_responses or self.DEFAULT_NORMAL_RESPONSES
        self._response_index = 0
        self._step_id = initial_step_id
        self._trajectory: List[ReasoningStep] = []
        self._active_failures: Dict[FailureMode, Dict[str, Any]] = {}
        self._failure_response_index: Dict[FailureMode, int] = {}

    def step(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> ReasoningStep:
        """
        Execute a single reasoning step.

        Args:
            prompt: The input prompt/task for this step.
            context: Optional context dictionary for additional metadata.

        Returns:
            ReasoningStep with the agent's response and failure mode info.
        """
        context = context or {}

        # Determine which failure modes are active
        active_modes = [fm for fm in self._active_failures if self._active_failures[fm].get("active", False)]

        if active_modes:
            # Pick the first active failure mode (priority order)
            failure_mode = active_modes[0]
            failure_config = self._active_failures[failure_mode]

            response = self._generate_failure_response(failure_mode, failure_config)
            step = ReasoningStep(
                step_id=self._step_id,
                content=response,
                failure_mode=failure_mode,
                response_mode=MockResponseMode.FAILURE_INJECTED,
                metadata={
                    "prompt": prompt,
                    "context": context,
                    "injection_reason": failure_config.get("reason", "injected_failure"),
                    "remaining_steps": failure_config.get("remaining_steps", 1),
                },
            )

            # Decrement remaining_steps, deactivate if exhausted
            if "remaining_steps" in failure_config:
                failure_config["remaining_steps"] -= 1
                if failure_config["remaining_steps"] <= 0:
                    failure_config["active"] = False
        else:
            # Normal response
            response = self._generate_normal_response()
            step = ReasoningStep(
                step_id=self._step_id,
                content=response,
                failure_mode=FailureMode.NONE,
                response_mode=MockResponseMode.NORMAL,
                metadata={"prompt": prompt, "context": context},
            )

        self._trajectory.append(step)
        self._step_id += 1
        return step

    def inject_failure(
        self,
        failure_mode: FailureMode,
        reason: str = "test_injection",
        remaining_steps: int = 1,
        probability: float = 1.0,
    ) -> None:
        """
        Inject a failure mode into the agent's reasoning.

        Args:
            failure_mode: The type of failure to inject.
            reason: Description of why the failure is being injected.
            remaining_steps: Number of steps the failure should remain active.
            probability: Probability of the failure triggering (0.0 to 1.0).
        """
        if failure_mode == FailureMode.NONE:
            raise ValueError("Cannot inject FailureMode.NONE as a failure mode.")

        self._active_failures[failure_mode] = {
            "active": True,
            "reason": reason,
            "remaining_steps": remaining_steps,
            "probability": probability,
        }

    def get_trajectory(self) -> List[ReasoningStep]:
        """
        Get the full reasoning trajectory.

        Returns:
            List of all ReasoningStep objects in order.
        """
        return list(self._trajectory)

    def get_trajectory_contents(self) -> List[str]:
        """
        Get just the text contents of the trajectory.

        Returns:
            List of reasoning step contents as strings.
        """
        return [step.content for step in self._trajectory]

    def clear_trajectory(self) -> None:
        """Clear the trajectory history but keep failure injections intact."""
        self._trajectory.clear()
        self._step_id = 0

    def reset(self) -> None:
        """Reset the agent to initial state, clearing trajectory and failures."""
        self._trajectory.clear()
        self._step_id = 0
        self._active_failures.clear()
        self._response_index = 0
        self._failure_response_index.clear()

    def deactivate_failure(self, failure_mode: FailureMode) -> None:
        """Manually deactivate a specific failure mode."""
        if failure_mode in self._active_failures:
            self._active_failures[failure_mode]["active"] = False

    def get_active_failures(self) -> List[FailureMode]:
        """Return list of currently active failure modes."""
        return [
            fm for fm, config in self._active_failures.items()
            if config.get("active", False)
        ]

    def _generate_normal_response(self) -> str:
        """Generate a normal reasoning response."""
        response = self._normal_responses[self._response_index % len(self._normal_responses)]
        self._response_index += 1
        return response

    def _generate_failure_response(
        self,
        failure_mode: FailureMode,
        config: Dict[str, Any],
    ) -> str:
        """Generate a failure-mode-specific response."""
        probability = config.get("probability", 1.0)
        import random
        if random.random() > probability:
            return self._generate_normal_response()

        responses = self.FAILURE_RESPONSES.get(failure_mode, self.DEFAULT_NORMAL_RESPONSES)
        if failure_mode not in self._failure_response_index:
            self._failure_response_index[failure_mode] = 0

        response = responses[self._failure_response_index[failure_mode] % len(responses)]
        self._failure_response_index[failure_mode] += 1
        return response

    def run_steps(
        self,
        prompts: List[str],
        inject_at: Optional[Dict[int, FailureMode]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ReasoningStep]:
        """
        Run multiple reasoning steps in sequence.

        Args:
            prompts: List of prompts, one per step.
            inject_at: Optional dict mapping step index to failure mode to inject.
            context: Optional context passed to each step.

        Returns:
            List of ReasoningStep objects from all steps.
        """
        inject_at = inject_at or {}
        results = []

        for i, prompt in enumerate(prompts):
            if i in inject_at:
                self.inject_failure(inject_at[i], reason=f"scheduled_injection_step_{i}")

            step = self.step(prompt, context)
            results.append(step)

        return results
