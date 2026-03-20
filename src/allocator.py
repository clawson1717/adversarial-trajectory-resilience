"""
Compute Allocator Module
Dynamic compute scaling based on uncertainty + failure risk.

This module provides:
- Budget management across trajectory branches
- Dynamic compute scaling based on uncertainty estimation
- Failure risk-based allocation decisions
- Termination recommendations for high-risk trajectories
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List, Set
import copy
import math


class BudgetStatus(Enum):
    """Status of budget allocation."""
    ALLOCATED = "allocated"
    EXHAUSTED = "exhausted"
    DEPLETED = "depleted"
    TERMINATED = "terminated"


@dataclass
class BranchBudget:
    """
    Compute budget for a single trajectory branch.
    
    Attributes:
        branch_id: Unique identifier for the branch
        allocated: Total compute units allocated
        consumed: Compute units already used
        remaining: Compute units remaining (allocated - consumed)
        priority: Branch priority (higher = more compute)
        risk_score: Current risk assessment (0.0-1.0)
        uncertainty: Current uncertainty estimate (0.0-1.0)
        metadata: Additional branch-specific data
    """
    branch_id: str
    allocated: float = 100.0
    consumed: float = 0.0
    priority: float = 1.0
    risk_score: float = 0.0
    uncertainty: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def remaining(self) -> float:
        """Compute remaining budget."""
        return max(0.0, self.allocated - self.consumed)
    
    @property
    def utilization(self) -> float:
        """Compute utilization ratio (0.0-1.0)."""
        if self.allocated <= 0:
            return 1.0
        return self.consumed / self.allocated
    
    @property
    def status(self) -> BudgetStatus:
        """Get current budget status."""
        if self.utilization >= 1.0:
            return BudgetStatus.DEPLETED
        elif self.utilization >= 0.9:
            return BudgetStatus.EXHAUSTED
        else:
            return BudgetStatus.ALLOCATED
    
    def consume(self, amount: float) -> float:
        """
        Consume compute budget.
        
        Args:
            amount: Amount to consume
            
        Returns:
            Actual amount consumed (may be less if budget insufficient)
        """
        actual = min(amount, self.remaining)
        self.consumed += actual
        return actual


@dataclass
class AllocationResult:
    """
    Result of a budget allocation decision.
    
    Attributes:
        branch_id: The branch that was allocated
        amount: Amount allocated
        success: Whether allocation succeeded
        reason: Human-readable reason for decision
        new_budget: Updated budget after allocation
    """
    branch_id: str
    amount: float
    success: bool
    reason: str
    new_budget: Optional[BranchBudget] = None


# Import existing components if available
try:
    from src.detector import FailureModeDetector, FailureMode, FailureClassification
except ImportError:
    FailureModeDetector = None
    FailureMode = None
    FailureClassification = None


class ComputeAllocator:
    """
    Dynamic compute allocation based on uncertainty and failure risk.
    
    This allocator manages compute budgets across trajectory branches,
    scaling allocation based on:
    - Uncertainty estimates (from UncertaintyEstimator)
    - Failure risk (from FailureModeDetector)
    - Branch priority and utilization
    
    Usage:
        allocator = ComputeAllocator(
            total_budget=1000.0,
            base_allocation=50.0,
            failure_mode_detector=detector
        )
        
        # Allocate compute for a branch
        result = allocator.allocate(branch_id, state)
        
        # Check termination
        if allocator.should_terminate(branch_id):
            print("Branch should be terminated")
        
        # Get current budget
        budget = allocator.get_budget(branch_id)
    """
    
    def __init__(
        self,
        total_budget: float = 1000.0,
        base_allocation: float = 50.0,
        min_allocation: float = 10.0,
        max_allocation: float = 200.0,
        uncertainty_threshold: float = 0.3,
        risk_threshold: float = 0.6,
        termination_threshold: float = 0.85,
        failure_mode_detector: Optional[Any] = None,
        uncertainty_estimator: Optional[Any] = None,
        conservative_mode: bool = False
    ):
        """
        Initialize the ComputeAllocator.
        
        Args:
            total_budget: Total compute budget across all branches
            base_allocation: Default allocation per branch
            min_allocation: Minimum allocation for any branch
            max_allocation: Maximum allocation for any branch
            uncertainty_threshold: Uncertainty above this triggers more compute
            risk_threshold: Risk above this reduces allocation
            termination_threshold: Risk above this triggers termination
            failure_mode_detector: Optional FailureModeDetector instance
            uncertainty_estimator: Optional UncertaintyEstimator instance
            conservative_mode: If True, use more aggressive termination
        """
        self.total_budget = total_budget
        self.base_allocation = base_allocation
        self.min_allocation = min_allocation
        self.max_allocation = max_allocation
        self.uncertainty_threshold = uncertainty_threshold
        self.risk_threshold = risk_threshold
        self.termination_threshold = termination_threshold
        self.conservative_mode = conservative_mode
        
        # Initialize components
        self.failure_mode_detector = failure_mode_detector
        self.uncertainty_estimator = uncertainty_estimator
        
        # Track budgets per branch
        self._branch_budgets: Dict[str, BranchBudget] = {}
        
        # Track global consumption
        self._total_consumed: float = 0.0
        
        # Allocation history for debugging
        self._allocation_history: List[AllocationResult] = []
    
    def allocate(
        self,
        branch_id: str,
        state: Optional[Dict[str, Any]] = None,
        requested_amount: Optional[float] = None
    ) -> AllocationResult:
        """
        Allocate compute budget for a trajectory branch.
        
        Args:
            branch_id: Unique identifier for the branch
            state: Optional state dictionary for risk/uncertainty assessment
            requested_amount: Optional explicit amount to allocate
            
        Returns:
            AllocationResult with allocation details
        """
        # Get or create branch budget
        budget = self._branch_budgets.get(branch_id)
        if budget is None:
            budget = BranchBudget(branch_id=branch_id, allocated=self.base_allocation)
            self._branch_budgets[branch_id] = budget
        
        # Calculate dynamic allocation based on state
        if state is not None:
            self._update_risk_uncertainty(budget, state)
        
        # Determine allocation amount
        if requested_amount is not None:
            amount = requested_amount
        else:
            amount = self._calculate_allocation(budget)
        
        # Check if we have global budget
        global_remaining = self.total_budget - self._total_consumed
        if global_remaining <= 0:
            result = AllocationResult(
                branch_id=branch_id,
                amount=0.0,
                success=False,
                reason="Global budget exhausted",
                new_budget=budget
            )
            self._allocation_history.append(result)
            return result
        
        # Adjust amount if exceeds global remaining
        if amount > global_remaining:
            amount = global_remaining
        
        # Cap at branch remaining budget
        branch_remaining = budget.remaining
        if amount > branch_remaining:
            if branch_remaining <= 0:
                result = AllocationResult(
                    branch_id=branch_id,
                    amount=0.0,
                    success=False,
                    reason="Branch budget depleted",
                    new_budget=budget
                )
                self._allocation_history.append(result)
                return result
            amount = branch_remaining
        
        # Perform allocation
        actual = budget.consume(amount)
        self._total_consumed += actual
        
        result = AllocationResult(
            branch_id=branch_id,
            amount=actual,
            success=actual > 0,
            reason=self._get_allocation_reason(budget),
            new_budget=budget
        )
        self._allocation_history.append(result)
        return result
    
    def get_budget(self, branch_id: str) -> Optional[BranchBudget]:
        """
        Get current budget for a branch.
        
        Args:
            branch_id: Branch identifier
            
        Returns:
            BranchBudget if branch exists, None otherwise
        """
        return self._branch_budgets.get(branch_id)
    
    def should_terminate(self, branch_id: str) -> bool:
        """
        Determine if a branch should be terminated.
        
        A branch should be terminated if:
        - Risk score exceeds termination threshold
        - Budget is depleted
        - Failure mode is critical
        
        Args:
            branch_id: Branch identifier
            
        Returns:
            True if branch should be terminated, False otherwise
        """
        budget = self._branch_budgets.get(branch_id)
        if budget is None:
            return True  # Unknown branch should be terminated
        
        # Check budget depletion
        if budget.status == BudgetStatus.DEPLETED:
            return True
        
        # Check risk threshold
        if budget.risk_score >= self.termination_threshold:
            return True
        
        # In conservative mode, also terminate on high utilization
        if self.conservative_mode:
            if budget.utilization >= 0.95:
                return True
        
        return False
    
    def get_termination_reason(self, branch_id: str) -> str:
        """
        Get the reason for termination recommendation.
        
        Args:
            branch_id: Branch identifier
            
        Returns:
            Human-readable reason for termination
        """
        budget = self._branch_budgets.get(branch_id)
        if budget is None:
            return "Unknown branch"
        
        if budget.status == BudgetStatus.DEPLETED:
            return f"Budget depleted (consumed {budget.consumed:.1f}/{budget.allocated:.1f})"
        
        if budget.risk_score >= self.termination_threshold:
            return f"Risk score {budget.risk_score:.2f} exceeds threshold {self.termination_threshold:.2f}"
        
        if budget.utilization >= 0.95:
            return f"Utilization {budget.utilization:.2f} is critically high"
        
        return "No termination reason found"
    
    def _update_risk_uncertainty(self, budget: BranchBudget, state: Dict[str, Any]) -> None:
        """Update risk and uncertainty scores from state."""
        content = state.get("content", "")
        trajectory_history = state.get("history", [])
        
        # Update uncertainty if estimator available
        if self.uncertainty_estimator is not None:
            try:
                budget.uncertainty = self.uncertainty_estimator.estimate_uncertainty(content)
            except Exception:
                # Fallback: estimate from content length and markers
                budget.uncertainty = self._estimate_uncertainty(content)
        else:
            budget.uncertainty = self._estimate_uncertainty(content)
        
        # Update risk from failure mode detector
        if self.failure_mode_detector is not None and FailureModeDetector is not None:
            try:
                classification = self.failure_mode_detector.classify_state(
                    content, trajectory_history
                )
                # Convert confidence to risk (inverse: high confidence = low risk)
                if classification.mode.value == "none":
                    budget.risk_score = 1.0 - classification.confidence
                else:
                    # High confidence in a failure mode = high risk
                    budget.risk_score = classification.confidence
                    # Adjust based on failure mode severity
                    budget.metadata["failure_mode"] = classification.mode.value
            except Exception:
                budget.risk_score = 0.0
        else:
            budget.risk_score = self._estimate_risk(content)
        
        # Update priority based on combined score
        budget.priority = self._calculate_priority(budget)
    
    def _estimate_uncertainty(self, content: str) -> float:
        """
        Estimate uncertainty from content when no estimator available.
        
        Uses heuristic markers:
        - Hedging language indicates uncertainty
        - Question marks indicate uncertainty
        - Longer content with qualifiers increases uncertainty
        """
        if not content:
            return 0.0
        
        uncertainty_markers = [
            "maybe", "perhaps", "might", "could be", "possibly",
            "not sure", "uncertain", "unclear", "ambiguous",
            "I think", "I believe", "I assume", "probably"
        ]
        
        content_lower = content.lower()
        marker_count = sum(1 for m in uncertainty_markers if m in content_lower)
        
        # Normalize to 0-1 range
        uncertainty = min(1.0, marker_count * 0.15)
        
        # Check for question marks
        if "?" in content:
            uncertainty += 0.1
        
        return min(1.0, uncertainty)
    
    def _estimate_risk(self, content: str) -> float:
        """
        Estimate risk from content when no detector available.
        
        Uses heuristic markers for potential failure modes.
        """
        if not content:
            return 0.0
        
        risk_markers = {
            "social_conformity": ["as you suggested", "you're right", "as you mentioned"],
            "self_doubt": ["I am not sure", "actually", "wait", "perhaps I was wrong"],
            "emotional": ["urgent", "immediately", "sorry", "apologize"],
            "suggestion_hijack": ["instead", "actually", "rather"]
        }
        
        content_lower = content.lower()
        risk_score = 0.0
        
        for mode, markers in risk_markers.items():
            if any(m in content_lower for m in markers):
                risk_score += 0.25
        
        return min(1.0, risk_score)
    
    def _calculate_priority(self, budget: BranchBudget) -> float:
        """
        Calculate branch priority based on risk and uncertainty.
        
        Higher priority = more compute should be allocated.
        
        Priority increases with:
        - Higher uncertainty (needs more exploration)
        - Lower risk (safe to continue)
        
        Priority decreases with:
        - Higher risk (may fail)
        - Higher utilization (already consumed much budget)
        """
        # Base priority from uncertainty
        uncertainty_factor = budget.uncertainty
        
        # Inverse risk factor (high risk = low priority)
        risk_factor = 1.0 - budget.risk_score
        
        # Utilization factor (high utilization = slightly lower priority)
        utilization_factor = 1.0 - (budget.utilization * 0.3)
        
        # Combined priority
        priority = (uncertainty_factor * 0.4 + risk_factor * 0.4 + utilization_factor * 0.2)
        
        return max(0.1, min(2.0, priority))
    
    def _calculate_allocation(self, budget: BranchBudget) -> float:
        """
        Calculate compute allocation for a branch.
        
        Scaling logic:
        - High uncertainty + low risk: Increase allocation (exploration)
        - High risk: Reduce allocation (conservative)
        - High uncertainty: Increase allocation (needs more compute)
        - Low remaining budget: Cap allocation
        """
        # Start with base allocation
        allocation = self.base_allocation
        
        # Scale by priority
        allocation *= budget.priority
        
        # Scale by uncertainty (high uncertainty = more compute)
        if budget.uncertainty > self.uncertainty_threshold:
            uncertainty_boost = 1.0 + (budget.uncertainty - self.uncertainty_threshold) * 1.5
            allocation *= uncertainty_boost
        
        # Scale by inverse risk (high risk = less compute)
        if budget.risk_score > self.risk_threshold:
            risk_reduction = 1.0 - (budget.risk_score - self.risk_threshold) * 1.2
            allocation *= max(0.2, risk_reduction)
        
        # Cap at min/max
        allocation = max(self.min_allocation, min(self.max_allocation, allocation))
        
        # Consider remaining budget
        if allocation > budget.remaining:
            allocation = budget.remaining * 0.5  # Don't consume all at once
        
        return allocation
    
    def _get_allocation_reason(self, budget: BranchBudget) -> str:
        """Generate human-readable reason for allocation."""
        reasons = []
        
        if budget.uncertainty > self.uncertainty_threshold:
            reasons.append(f"high uncertainty ({budget.uncertainty:.2f})")
        
        if budget.risk_score > self.risk_threshold:
            reasons.append(f"high risk ({budget.risk_score:.2f})")
        
        if budget.priority != 1.0:
            reasons.append(f"priority={budget.priority:.2f}")
        
        if budget.utilization > 0.5:
            reasons.append(f"utilization={budget.utilization:.2f}")
        
        if not reasons:
            return "Standard allocation"
        
        return f"Adjusted for: {', '.join(reasons)}"
    
    def rebalance(
        self,
        total_budget: Optional[float] = None,
        min_budget_per_branch: Optional[float] = None
    ) -> Dict[str, AllocationResult]:
        """
        Rebalance budgets across all branches.
        
        Args:
            total_budget: New total budget (optional)
            min_budget_per_branch: Minimum budget per branch (optional)
            
        Returns:
            Dictionary of branch_id -> AllocationResult for affected branches
        """
        if total_budget is not None:
            self.total_budget = total_budget
        
        if min_budget_per_branch is not None:
            self.min_allocation = min_budget_per_branch
        
        results = {}
        
        # Reallocate based on priorities
        total_priority = sum(b.priority for b in self._branch_budgets.values())
        
        for branch_id, budget in self._branch_budgets.items():
            if total_priority > 0:
                fair_share = (budget.priority / total_priority) * self.total_budget
                new_allocation = max(self.min_allocation, fair_share)
                
                # Only update if significantly different
                if abs(new_allocation - budget.allocated) > budget.allocated * 0.1:
                    budget.allocated = new_allocation
                    results[branch_id] = AllocationResult(
                        branch_id=branch_id,
                        amount=new_allocation - budget.consumed,
                        success=True,
                        reason="Rebalanced based on priorities",
                        new_budget=budget
                    )
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get overall allocator status.
        
        Returns:
            Dictionary with global status information
        """
        total_allocated = sum(b.allocated for b in self._branch_budgets.values())
        total_remaining = sum(b.remaining for b in self._branch_budgets.values())
        
        return {
            "total_budget": self.total_budget,
            "total_allocated": total_allocated,
            "total_consumed": self._total_consumed,
            "total_remaining": self.total_budget - self._total_consumed,
            "num_branches": len(self._branch_budgets),
            "branches": {
                branch_id: {
                    "allocated": b.allocated,
                    "consumed": b.consumed,
                    "remaining": b.remaining,
                    "utilization": b.utilization,
                    "priority": b.priority,
                    "risk_score": b.risk_score,
                    "uncertainty": b.uncertainty,
                    "status": b.status.value
                }
                for branch_id, b in self._branch_budgets.items()
            }
        }
    
    def reset_branch(self, branch_id: str) -> bool:
        """
        Reset budget for a branch to initial state.
        
        Args:
            branch_id: Branch identifier
            
        Returns:
            True if branch was reset, False if branch didn't exist
        """
        if branch_id in self._branch_budgets:
            budget = self._branch_budgets[branch_id]
            self._total_consumed -= budget.consumed
            budget.consumed = 0.0
            budget.risk_score = 0.0
            budget.uncertainty = 0.0
            budget.metadata.clear()
            return True
        return False
    
    def remove_branch(self, branch_id: str) -> bool:
        """
        Remove a branch and reclaim its budget.
        
        Args:
            branch_id: Branch identifier
            
        Returns:
            True if branch was removed, False if branch didn't exist
        """
        if branch_id in self._branch_budgets:
            budget = self._branch_budgets.pop(branch_id)
            self._total_consumed -= budget.consumed
            return True
        return False
    
    def get_allocation_history(
        self,
        branch_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[AllocationResult]:
        """
        Get allocation history.
        
        Args:
            branch_id: Optional filter by branch
            limit: Maximum number of results to return
            
        Returns:
            List of AllocationResult objects
        """
        history = self._allocation_history
        
        if branch_id is not None:
            history = [r for r in history if r.branch_id == branch_id]
        
        if limit is not None:
            history = history[-limit:]
        
        return history
