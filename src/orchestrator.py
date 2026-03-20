"""
Resilience Orchestrator Module
Main pipeline coordinator for adversarial trajectory resilience.

Coordinates all components into a unified pipeline:
    monitor → detect → prune → allocate → verify

State machine for trajectory health management.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List, Set, Callable
from datetime import datetime, timezone
import copy
import json

# Import existing components
from src.trajectory import TrajectoryGraph
from src.detector import FailureModeDetector, FailureMode, FailureClassification
from src.verification import VerificationGate, VerificationResult, Checkpoint, VerificationError
from src.allocator import ComputeAllocator, BranchBudget, BudgetStatus


class TrajectoryHealth(Enum):
    """Health states for trajectory state machine."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    RECOVERING = "recovering"
    TERMINATED = "terminated"


class PipelineStage(Enum):
    """Pipeline execution stages."""
    MONITOR = "monitor"
    DETECT = "detect"
    PRUNE = "prune"
    ALLOCATE = "allocate"
    VERIFY = "verify"
    IDLE = "idle"


@dataclass
class StageResult:
    """Result from executing a pipeline stage."""
    stage: PipelineStage
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    duration_ms: float = 0.0

    def __bool__(self) -> bool:
        return self.success


@dataclass
class OrchestrationState:
    """Current state of the orchestration."""
    health: TrajectoryHealth = TrajectoryHealth.HEALTHY
    current_stage: PipelineStage = PipelineStage.IDLE
    active_branch_id: Optional[str] = None
    cycle_count: int = 0
    last_verification_passed: bool = True
    consecutive_failures: int = 0
    total_states_processed: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageHistoryEntry:
    """Historical record of a stage execution."""
    timestamp: str
    stage: str
    branch_id: str
    success: bool
    duration_ms: float
    summary: str


class ResilienceOrchestrator:
    """
    Main pipeline coordinator for adversarial trajectory resilience.

    Orchestrates the complete pipeline:
        monitor → detect → prune → allocate → verify

    Manages trajectory health via state machine transitions.

    Usage:
        orchestrator = ResilienceOrchestrator()

        # Initialize with components
        orchestrator.initialize(
            trajectory_graph=graph,
            failure_mode_detector=detector,
            verification_gate=gate,
            compute_allocator=allocator
        )

        # Step through pipeline for a state
        result = orchestrator.step(state, branch_id)

        # Or run full pipeline
        result = orchestrator.run_pipeline(state, branch_id)

        # Check trajectory health
        health = orchestrator.get_health(branch_id)

        # Rollback if needed
        if orchestrator.needs_rollback(branch_id):
            state = orchestrator.rollback_to_safe(branch_id)
    """

    def __init__(
        self,
        uncertainty_threshold: float = 0.3,
        health_transition_callback: Optional[Callable[[str, TrajectoryHealth, TrajectoryHealth], None]] = None,
        auto_initialize: bool = False
    ):
        """
        Initialize the ResilienceOrchestrator.

        Args:
            uncertainty_threshold: Threshold for uncertainty estimation
            health_transition_callback: Optional callback for health transitions
            auto_initialize: If True, create default components on first step
        """
        self.uncertainty_threshold = uncertainty_threshold
        self.health_transition_callback = health_transition_callback
        self.auto_initialize = auto_initialize

        # Components (set via initialize or auto-init)
        self._trajectory_graph: Optional[TrajectoryGraph] = None
        self._failure_mode_detector: Optional[FailureModeDetector] = None
        self._uncertainty_estimator: Optional[Any] = None  # Set if provided
        self._verification_gate: Optional[VerificationGate] = None
        self._compute_allocator: Optional[ComputeAllocator] = None

        # State management
        self._orchestration_states: Dict[str, OrchestrationState] = {}
        self._stage_history: List[StageHistoryEntry] = []
        self._branch_metadata: Dict[str, Dict[str, Any]] = {}

        # Pipeline configuration
        self._stage_handlers: Dict[PipelineStage, Callable] = {
            PipelineStage.MONITOR: self._stage_monitor,
            PipelineStage.DETECT: self._stage_detect,
            PipelineStage.PRUNE: self._stage_prune,
            PipelineStage.ALLOCATE: self._stage_allocate,
            PipelineStage.VERIFY: self._stage_verify,
        }

        # Trajectory pruning callback (for TrajectoryPruner-like functionality)
        self._prune_condition: Optional[Callable[[str, Dict], bool]] = None

        # Callbacks
        self._on_stage_complete: Dict[PipelineStage, List[Callable]] = {
            stage: [] for stage in PipelineStage
        }
        self._on_health_change: Dict[str, List[Callable]] = {}

        # Statistics
        self._stats = {
            "total_pipeline_runs": 0,
            "total_stage_executions": 0,
            "total_health_transitions": 0,
            "total_rollbacks": 0,
        }

    def initialize(
        self,
        trajectory_graph: Optional[TrajectoryGraph] = None,
        failure_mode_detector: Optional[FailureModeDetector] = None,
        uncertainty_estimator: Optional[Any] = None,
        verification_gate: Optional[VerificationGate] = None,
        compute_allocator: Optional[ComputeAllocator] = None
    ) -> None:
        """
        Initialize orchestrator with components.

        Args:
            trajectory_graph: TrajectoryGraph instance
            failure_mode_detector: FailureModeDetector instance
            uncertainty_estimator: UncertaintyEstimator instance (optional)
            verification_gate: VerificationGate instance
            compute_allocator: ComputeAllocator instance
        """
        self._trajectory_graph = trajectory_graph or TrajectoryGraph()
        self._failure_mode_detector = failure_mode_detector or FailureModeDetector()
        self._uncertainty_estimator = uncertainty_estimator
        self._verification_gate = verification_gate or VerificationGate(
            failure_mode_detector=self._failure_mode_detector,
            uncertainty_estimator=self._uncertainty_estimator
        )
        self._compute_allocator = compute_allocator or ComputeAllocator(
            failure_mode_detector=self._failure_mode_detector,
            uncertainty_estimator=self._uncertainty_estimator
        )

    @property
    def is_initialized(self) -> bool:
        """Check if orchestrator has all required components."""
        return (
            self._trajectory_graph is not None
            and self._failure_mode_detector is not None
            and self._verification_gate is not None
            and self._compute_allocator is not None
        )

    def ensure_initialized(self) -> None:
        """Ensure orchestrator is initialized, raising if not."""
        if not self.is_initialized:
            if self.auto_initialize:
                self.initialize()
            else:
                raise RuntimeError(
                    "ResilienceOrchestrator not initialized. "
                    "Call initialize() or set auto_initialize=True."
                )

    def step(
        self,
        state: Dict[str, Any],
        branch_id: str,
        stage: Optional[PipelineStage] = None
    ) -> StageResult:
        """
        Execute a single pipeline stage.

        Args:
            state: Current state dictionary
            branch_id: Branch identifier
            stage: Stage to execute (defaults to next logical stage)

        Returns:
            StageResult from the executed stage
        """
        self.ensure_initialized()

        # Ensure state exists
        self._ensure_branch_state(branch_id)

        # Determine stage
        if stage is None:
            stage = self._get_next_stage(branch_id)

        # Execute stage
        start_time = datetime.now(timezone.utc)
        handler = self._stage_handlers.get(stage, self._stage_idle)
        result = handler(state, branch_id)
        duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        result.duration_ms = duration_ms

        # Update orchestration state
        self._update_state_after_stage(branch_id, stage, result)

        # Record history
        self._record_stage_history(branch_id, stage, result, duration_ms)

        # Update stats
        self._stats["total_stage_executions"] += 1

        # Fire callbacks
        self._fire_stage_callbacks(stage, branch_id, result)

        return result

    def run_pipeline(
        self,
        state: Dict[str, Any],
        branch_id: str,
        stages: Optional[List[PipelineStage]] = None
    ) -> List[StageResult]:
        """
        Run the complete pipeline or specified stages.

        Args:
            state: Current state dictionary
            branch_id: Branch identifier
            stages: List of stages to run (defaults to full pipeline)

        Returns:
            List of StageResult for each executed stage
        """
        self.ensure_initialized()

        if stages is None:
            stages = [
                PipelineStage.MONITOR,
                PipelineStage.DETECT,
                PipelineStage.PRUNE,
                PipelineStage.ALLOCATE,
                PipelineStage.VERIFY,
            ]

        results = []
        for stage in stages:
            result = self.step(state, branch_id, stage)
            results.append(result)

            # Stop pipeline on critical failure
            if not result.success and stage == PipelineStage.VERIFY:
                # Check if it's a critical failure
                if result.errors:
                    break

        self._stats["total_pipeline_runs"] += 1
        return results

    def _stage_monitor(self, state: Dict[str, Any], branch_id: str) -> StageResult:
        """
        MONITOR stage: Collect state metrics and update trajectory graph.

        - Add/update node in trajectory graph
        - Update branch metadata with current state info
        - Check for cycles and dead ends
        """
        result = StageResult(stage=PipelineStage.MONITOR, success=True)
        errors = []

        try:
            state_id = state.get("id", f"state_{branch_id}_{self._get_state_count(branch_id)}")

            # Add node to trajectory graph
            self._trajectory_graph.add_node(state_id, metadata={
                "branch_id": branch_id,
                "content_preview": state.get("content", "")[:100],
                "confidence": state.get("confidence", 1.0),
            })

            # Update branch metadata
            if branch_id not in self._branch_metadata:
                self._branch_metadata[branch_id] = {}
            self._branch_metadata[branch_id]["last_state_id"] = state_id
            self._branch_metadata[branch_id]["state_count"] = \
                self._branch_metadata[branch_id].get("state_count", 0) + 1

            # Check for cycles
            cycles = self._trajectory_graph.detect_cycles()
            if cycles:
                result.data["cycles_detected"] = cycles
                self._transition_health(branch_id, TrajectoryHealth.DEGRADED)
                errors.append(f"Cycles detected: {len(cycles)}")

            # Check for dead ends (only if trajectory has nodes beyond this one)
            dead_ends = self._trajectory_graph.find_dead_ends()
            if state_id in dead_ends and len(self._trajectory_graph.nodes) > 1:
                result.data["is_dead_end"] = True
                self._transition_health(branch_id, TrajectoryHealth.CRITICAL)
                errors.append("Dead end reached")

            # Update uncertainty estimate
            content = state.get("content", "")
            uncertainty = self._estimate_uncertainty(content)
            result.data["uncertainty"] = uncertainty

            if branch_id in self._orchestration_states:
                self._orchestration_states[branch_id].total_states_processed += 1

        except Exception as e:
            result.success = False
            errors.append(f"Monitor error: {str(e)}")

        result.errors = errors
        return result

    def _stage_detect(self, state: Dict[str, Any], branch_id: str) -> StageResult:
        """
        DETECT stage: Classify failure modes using FailureModeDetector.

        - Run failure mode classification
        - Update risk scores
        - Determine if trajectory should continue
        """
        result = StageResult(stage=PipelineStage.DETECT, success=True)
        errors = []

        try:
            content = state.get("content", "")
            history = state.get("history", [])

            # Classify failure mode
            classification = self._failure_mode_detector.classify_state(content, history)

            result.data["failure_classification"] = {
                "mode": classification.mode.value,
                "confidence": classification.confidence,
                "metadata": classification.metadata,
            }

            # Update branch risk score
            if classification.mode != FailureMode.NONE:
                result.data["failure_detected"] = True

                # Determine severity
                if classification.confidence > 0.75:
                    self._transition_health(branch_id, TrajectoryHealth.CRITICAL)
                    errors.append(f"High-confidence failure mode: {classification.mode.value}")
                    result.success = False
                elif classification.confidence > 0.5:
                    self._transition_health(branch_id, TrajectoryHealth.DEGRADED)
                    errors.append(f"Potential failure mode: {classification.mode.value}")

                # Update compute allocator with risk
                if self._compute_allocator:
                    budget = self._compute_allocator.get_budget(branch_id)
                    if budget:
                        budget.metadata["failure_mode"] = classification.mode.value
                        budget.risk_score = max(budget.risk_score, classification.confidence)

            # Update compute allocation with uncertainty
            if self._uncertainty_estimator:
                uncertainty = self._uncertainty_estimator.estimate_uncertainty(content)
            else:
                uncertainty = self._estimate_uncertainty(content)

            result.data["uncertainty"] = uncertainty

        except Exception as e:
            result.success = False
            errors.append(f"Detection error: {str(e)}")

        result.errors = errors
        return result

    def _stage_prune(self, state: Dict[str, Any], branch_id: str) -> StageResult:
        """
        PRUNE stage: Identify and mark trajectories for pruning.

        - Check prune conditions
        - Mark unhealthy trajectories
        - Update trajectory graph
        """
        result = StageResult(stage=PipelineStage.PRUNE, success=True)
        errors = []

        try:
            pruned = False
            prune_reason = None

            # Check health-based pruning
            health = self.get_health(branch_id)
            if health == TrajectoryHealth.TERMINATED:
                pruned = True
                prune_reason = "Health terminated"
            elif health == TrajectoryHealth.CRITICAL:
                # Check if we should prune
                if self._should_prune_critical(branch_id):
                    pruned = True
                    prune_reason = "Critical health with high risk"

            # Check custom prune conditions
            if not pruned and self._prune_condition:
                if self._prune_condition(branch_id, state):
                    pruned = True
                    prune_reason = "Custom prune condition"

            # Check cycle-based pruning
            if not pruned:
                cycles = self._trajectory_graph.detect_cycles()
                for cycle in cycles:
                    if state.get("id") in cycle or any(
                        self._branch_metadata.get(branch_id, {}).get("last_state_id") == node
                        for node in cycle[:-1]
                    ):
                        # Check if this cycle involves the current branch
                        pruned = True
                        prune_reason = f"Cycle detected: {' -> '.join(cycle[-3:])}"
                        break

            result.data["pruned"] = pruned
            result.data["prune_reason"] = prune_reason

            if pruned:
                self._transition_health(branch_id, TrajectoryHealth.TERMINATED)
                result.data["action"] = "terminated"
            else:
                result.data["action"] = "continued"

        except Exception as e:
            result.success = False
            errors.append(f"Prune error: {str(e)}")

        result.errors = errors
        return result

    def _stage_allocate(self, state: Dict[str, Any], branch_id: str) -> StageResult:
        """
        ALLOCATE stage: Allocate compute budget based on state assessment.

        - Update risk/uncertainty in allocator
        - Request compute allocation
        - Check termination conditions
        """
        result = StageResult(stage=PipelineStage.ALLOCATE, success=True)
        errors = []

        try:
            # Get or create branch budget
            budget = self._compute_allocator.get_budget(branch_id)
            if budget is None:
                allocation = self._compute_allocator.allocate(branch_id, state)
                budget = allocation.new_budget
            else:
                # Update allocation based on current state
                allocation = self._compute_allocator.allocate(branch_id, state)

            result.data["allocation"] = {
                "amount": allocation.amount,
                "success": allocation.success,
                "reason": allocation.reason,
                "branch_allocated": budget.allocated if budget else 0,
                "branch_consumed": budget.consumed if budget else 0,
                "branch_remaining": budget.remaining if budget else 0,
            }

            # Check termination
            should_terminate = self._compute_allocator.should_terminate(branch_id)
            result.data["should_terminate"] = should_terminate

            if should_terminate:
                termination_reason = self._compute_allocator.get_termination_reason(branch_id)
                result.data["termination_reason"] = termination_reason
                errors.append(f"Termination recommended: {termination_reason}")

                # Transition to terminated if allocator says so
                if budget and budget.status == BudgetStatus.DEPLETED:
                    self._transition_health(branch_id, TrajectoryHealth.TERMINATED)
                    result.success = False

        except Exception as e:
            result.success = False
            errors.append(f"Allocation error: {str(e)}")

        result.errors = errors
        return result

    def _stage_verify(self, state: Dict[str, Any], branch_id: str) -> StageResult:
        """
        VERIFY stage: Verify state using VerificationGate.

        - Run verification checks
        - Create checkpoint if verified
        - Determine if state is safe to commit
        """
        result = StageResult(stage=PipelineStage.VERIFY, success=True)
        errors = []

        try:
            # Run verification
            verification = self._verification_gate.verify_transition(state)

            result.data["verification"] = verification.get_summary()
            result.data["verification_details"] = {
                "passed": verification.passed,
                "total_checks": len(verification.checks),
                "critical_failures": [
                    c.name for c in verification.checks
                    if c.result.value == "fail" and c.severity == "critical"
                ],
            }

            # Update orchestration state
            orch_state = self._get_or_create_state(branch_id)
            orch_state.last_verification_passed = verification.passed

            if not verification.passed:
                orch_state.consecutive_failures += 1
                critical = result.data["verification_details"]["critical_failures"]
                errors.append(f"Verification failed: {critical}")

                if orch_state.consecutive_failures >= 3:
                    self._transition_health(branch_id, TrajectoryHealth.CRITICAL)
                    result.success = False
            else:
                orch_state.consecutive_failures = 0

            # Create checkpoint on successful verification
            if verification.passed:
                try:
                    checkpoint = self._verification_gate.create_checkpoint(state)
                    result.data["checkpoint_id"] = checkpoint.checkpoint_id
                except (ValueError, VerificationError) as e:
                    errors.append(f"Checkpoint creation failed: {str(e)}")

            # Add verification details to result
            result.data["verification_timestamp"] = verification.timestamp

        except Exception as e:
            result.success = False
            errors.append(f"Verification error: {str(e)}")

        result.errors = errors
        return result

    def _stage_idle(self, state: Dict[str, Any], branch_id: str) -> StageResult:
        """Idle stage - no-op."""
        return StageResult(
            stage=PipelineStage.IDLE,
            success=True,
            data={"message": "Idle stage"}
        )

    def _get_next_stage(self, branch_id: str) -> PipelineStage:
        """Determine the next logical stage based on current state."""
        state = self._orchestration_states.get(branch_id)
        if state is None:
            return PipelineStage.MONITOR

        # Simple round-robin through stages
        stage_order = [
            PipelineStage.MONITOR,
            PipelineStage.DETECT,
            PipelineStage.PRUNE,
            PipelineStage.ALLOCATE,
            PipelineStage.VERIFY,
        ]

        current_idx = -1
        if state.current_stage in stage_order:
            current_idx = stage_order.index(state.current_stage)

        next_idx = (current_idx + 1) % len(stage_order)
        return stage_order[next_idx]

    def _ensure_branch_state(self, branch_id: str) -> None:
        """Ensure branch state exists."""
        if branch_id not in self._orchestration_states:
            self._orchestration_states[branch_id] = OrchestrationState(
                health=TrajectoryHealth.HEALTHY,
                current_stage=PipelineStage.IDLE,
                active_branch_id=branch_id,
            )

    def _get_or_create_state(self, branch_id: str) -> OrchestrationState:
        """Get or create orchestration state for a branch."""
        self._ensure_branch_state(branch_id)
        return self._orchestration_states[branch_id]

    def _get_state_count(self, branch_id: str) -> int:
        """Get the number of states processed for a branch."""
        return self._branch_metadata.get(branch_id, {}).get("state_count", 0)

    def _transition_health(
        self,
        branch_id: str,
        new_health: TrajectoryHealth
    ) -> None:
        """
        Transition branch to new health state.

        Args:
            branch_id: Branch identifier
            new_health: New health state
        """
        state = self._get_or_create_state(branch_id)
        old_health = state.health

        if old_health == new_health:
            return

        state.health = new_health
        self._stats["total_health_transitions"] += 1

        # Fire transition callback
        if self.health_transition_callback:
            try:
                self.health_transition_callback(branch_id, old_health, new_health)
            except Exception:
                pass

        # Fire branch-specific callbacks
        if branch_id in self._on_health_change:
            for callback in self._on_health_change[branch_id]:
                try:
                    callback(branch_id, old_health, new_health)
                except Exception:
                    pass

    def _should_prune_critical(self, branch_id: str) -> bool:
        """Determine if a critical branch should be pruned."""
        state = self._orchestration_states.get(branch_id)
        if state is None:
            return True

        # Prune if in critical state for too long
        if state.cycle_count > 10:
            return True

        # Prune if too many consecutive failures
        if state.consecutive_failures >= 5:
            return True

        return False

    def _estimate_uncertainty(self, content: str) -> float:
        """Estimate uncertainty when no estimator available."""
        if not content:
            return 0.0

        uncertainty_markers = [
            "maybe", "perhaps", "might", "could be", "possibly",
            "not sure", "uncertain", "unclear", "ambiguous",
            "I think", "I believe", "I assume", "probably"
        ]

        content_lower = content.lower()
        marker_count = sum(1 for m in uncertainty_markers if m in content_lower)
        uncertainty = min(1.0, marker_count * 0.15)

        if "?" in content:
            uncertainty += 0.1

        return min(1.0, uncertainty)

    def _update_state_after_stage(
        self,
        branch_id: str,
        stage: PipelineStage,
        result: StageResult
    ) -> None:
        """Update orchestration state after stage execution."""
        state = self._get_or_create_state(branch_id)
        state.current_stage = stage

        if stage == PipelineStage.VERIFY:
            state.cycle_count += 1

    def _record_stage_history(
        self,
        branch_id: str,
        stage: PipelineStage,
        result: StageResult,
        duration_ms: float
    ) -> None:
        """Record stage execution in history."""
        entry = StageHistoryEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            stage=stage.value,
            branch_id=branch_id,
            success=result.success,
            duration_ms=duration_ms,
            summary=f"{stage.value}: {'passed' if result.success else 'failed'}"
        )
        self._stage_history.append(entry)

        # Keep history bounded
        if len(self._stage_history) > 1000:
            self._stage_history = self._stage_history[-500:]

    def _fire_stage_callbacks(
        self,
        stage: PipelineStage,
        branch_id: str,
        result: StageResult
    ) -> None:
        """Fire registered callbacks for a stage."""
        for callback in self._on_stage_complete.get(stage, []):
            try:
                callback(branch_id, result)
            except Exception:
                pass

    # Public API

    def get_health(self, branch_id: str) -> TrajectoryHealth:
        """
        Get current health state for a branch.

        Args:
            branch_id: Branch identifier

        Returns:
            Current TrajectoryHealth state
        """
        state = self._orchestration_states.get(branch_id)
        return state.health if state else TrajectoryHealth.HEALTHY

    def get_state(self, branch_id: str) -> Optional[OrchestrationState]:
        """
        Get full orchestration state for a branch.

        Args:
            branch_id: Branch identifier

        Returns:
            OrchestrationState if branch exists, None otherwise
        """
        return self._orchestration_states.get(branch_id)

    def needs_rollback(self, branch_id: str) -> bool:
        """
        Check if a branch needs rollback.

        Args:
            branch_id: Branch identifier

        Returns:
            True if rollback is recommended
        """
        health = self.get_health(branch_id)
        return health in (TrajectoryHealth.CRITICAL, TrajectoryHealth.TERMINATED)

    def rollback_to_safe(
        self,
        branch_id: str,
        checkpoint_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Rollback to a safe checkpoint.

        Args:
            branch_id: Branch identifier
            checkpoint_id: Specific checkpoint to restore (defaults to latest)

        Returns:
            State data if rollback successful, None otherwise
        """
        if not self._verification_gate:
            return None

        if checkpoint_id:
            state = self._verification_gate.rollback_to_checkpoint(checkpoint_id)
        else:
            # Try to get latest checkpoint
            checkpoint = self._verification_gate.get_latest_checkpoint()
            if checkpoint:
                state = checkpoint.state_data
            else:
                state = None

        if state:
            self._stats["total_rollbacks"] += 1
            self._transition_health(branch_id, TrajectoryHealth.RECOVERING)

        return state

    def register_stage_callback(
        self,
        stage: PipelineStage,
        callback: Callable[[str, StageResult], None]
    ) -> None:
        """Register a callback for stage completion."""
        if stage not in self._on_stage_complete:
            self._on_stage_complete[stage] = []
        self._on_stage_complete[stage].append(callback)

    def register_health_callback(
        self,
        branch_id: str,
        callback: Callable[[str, TrajectoryHealth, TrajectoryHealth], None]
    ) -> None:
        """Register a callback for health transitions."""
        if branch_id not in self._on_health_change:
            self._on_health_change[branch_id] = []
        self._on_health_change[branch_id].append(callback)

    def set_prune_condition(
        self,
        condition: Callable[[str, Dict], bool]
    ) -> None:
        """Set custom prune condition function."""
        self._prune_condition = condition

    def get_stage_history(
        self,
        branch_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[StageHistoryEntry]:
        """Get stage execution history."""
        history = self._stage_history

        if branch_id is not None:
            history = [h for h in history if h.branch_id == branch_id]

        if limit is not None:
            history = history[-limit:]

        return history

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return copy.deepcopy(self._stats)

    def get_all_health(self) -> Dict[str, TrajectoryHealth]:
        """Get health states for all branches."""
        return {
            branch_id: state.health
            for branch_id, state in self._orchestration_states.items()
        }

    def export_state(self) -> str:
        """Export current orchestration state as JSON."""
        state_dict = {
            "orchestrator": {
                "stats": self._stats,
                "num_branches": len(self._orchestration_states),
            },
            "branches": {
                branch_id: {
                    "health": state.health.value,
                    "current_stage": state.current_stage.value,
                    "cycle_count": state.cycle_count,
                    "last_verification_passed": state.last_verification_passed,
                    "consecutive_failures": state.consecutive_failures,
                    "total_states_processed": state.total_states_processed,
                }
                for branch_id, state in self._orchestration_states.items()
            },
            "stage_history_count": len(self._stage_history),
        }
        return json.dumps(state_dict, indent=2)

    def reset(self) -> None:
        """Reset orchestrator to initial state."""
        self._orchestration_states.clear()
        self._stage_history.clear()
        self._branch_metadata.clear()
        self._stats = {
            "total_pipeline_runs": 0,
            "total_stage_executions": 0,
            "total_health_transitions": 0,
            "total_rollbacks": 0,
        }
