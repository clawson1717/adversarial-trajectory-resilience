"""
Adversarial Trajectory Resilience Package
"""
__version__ = "0.1.0"

from src.trajectory import TrajectoryGraph
from src.detector import FailureModeDetector, FailureMode, FailureClassification
from src.verification import VerificationGate, VerificationResult, Checkpoint, VerificationError
from src.allocator import ComputeAllocator, BranchBudget, BudgetStatus, AllocationResult
from src.orchestrator import (
    ResilienceOrchestrator,
    TrajectoryHealth,
    PipelineStage,
    StageResult,
    OrchestrationState,
    StageHistoryEntry,
)
