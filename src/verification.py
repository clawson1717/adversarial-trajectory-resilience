"""
Verification Gate Module
Inspired by LawThinker: enforces verification after critical steps.

This module provides a safety layer that:
- Performs binary checklist evaluation for each state transition
- Verifies logical consistency of transitions
- Checks confidence thresholds from UncertaintyEstimator
- Validates failure mode severity from FailureModeDetector
- Creates immutable checkpoints for state recovery
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timezone
import copy
import hashlib
import json

# Import existing components if available
try:
    from src.detector import FailureModeDetector, FailureMode, FailureClassification
except ImportError:
    FailureModeDetector = None
    FailureMode = None
    FailureClassification = None


class CheckResult(Enum):
    """Binary result for a single check."""
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class CheckItem:
    """A single verification check item."""
    name: str
    description: str
    result: CheckResult = CheckResult.SKIP
    details: str = ""
    severity: str = "info"  # info, warning, critical


@dataclass
class VerificationResult:
    """
    Result of verification with binary checklist evaluation.
    
    Attributes:
        passed: Overall pass/fail status
        checks: List of individual check items
        timestamp: When verification occurred
        state_id: Identifier of the verified state
        transition_id: Identifier of the transition if applicable
        details: Additional context about the verification
    """
    passed: bool
    checks: List[CheckItem] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    state_id: Optional[str] = None
    transition_id: Optional[str] = None
    details: str = ""

    def add_check(self, name: str, description: str, result: CheckResult, 
                  details: str = "", severity: str = "info"):
        """Add a check item to the verification result."""
        self.checks.append(CheckItem(
            name=name,
            description=description,
            result=result,
            details=details,
            severity=severity
        ))
        
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the verification result."""
        return {
            "passed": self.passed,
            "total_checks": len(self.checks),
            "passed_checks": sum(1 for c in self.checks if c.result == CheckResult.PASS),
            "failed_checks": sum(1 for c in self.checks if c.result == CheckResult.FAIL),
            "skipped_checks": sum(1 for c in self.checks if c.result == CheckResult.SKIP),
            "critical_failures": [c.name for c in self.checks 
                                  if c.result == CheckResult.FAIL and c.severity == "critical"]
        }


@dataclass
class Checkpoint:
    """
    Immutable checkpoint of a state.
    
    Once created, a checkpoint should not be modified. It serves as
    a recovery point and audit trail for state transitions.
    """
    checkpoint_id: str
    state_id: str
    state_data: Dict[str, Any]
    created_at: str
    checksum: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def verify_integrity(self) -> bool:
        """Verify the checkpoint has not been tampered with."""
        current_checksum = self._compute_checksum()
        return current_checksum == self.checksum
    
    def _compute_checksum(self) -> str:
        """Compute SHA-256 checksum of the state data."""
        state_json = json.dumps(self.state_data, sort_keys=True)
        return hashlib.sha256(state_json.encode()).hexdigest()


class VerificationGate:
    """
    LawThinker-inspired verification gate for state transitions.
    
    Adds a safety layer before committing to paths by performing
    binary checklist evaluation for each state transition.
    
    Usage:
        gate = VerificationGate()
        
        # Verify a transition
        result = gate.verify_transition(state, transition)
        if not result.passed:
            raise VerificationError(result.details)
        
        # Create checkpoint before committing
        checkpoint = gate.create_checkpoint(state)
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.7,
        max_severity: str = "critical",
        failure_mode_detector: Optional[Any] = None,
        uncertainty_estimator: Optional[Any] = None,
        custom_checks: Optional[List[Callable]] = None
    ):
        """
        Initialize the VerificationGate.
        
        Args:
            confidence_threshold: Minimum confidence to pass (0.0-1.0)
            max_severity: Maximum allowed failure severity ("info", "warning", "critical")
            failure_mode_detector: Optional FailureModeDetector instance
            uncertainty_estimator: Optional UncertaintyEstimator instance
            custom_checks: Optional list of custom verification functions
        """
        self.confidence_threshold = confidence_threshold
        self.max_severity = max_severity
        self.failure_mode_detector = failure_mode_detector
        self.uncertainty_estimator = uncertainty_estimator
        self.custom_checks = custom_checks or []
        
        # Checkpoint storage
        self._checkpoints: Dict[str, Checkpoint] = {}
        self._checkpoint_history: List[str] = []
    
    def verify_transition(
        self, 
        state: Dict[str, Any], 
        transition: Optional[Dict[str, Any]] = None
    ) -> VerificationResult:
        """
        Verify a state transition with binary checklist evaluation.
        
        Args:
            state: Current state dictionary
            transition: Optional transition information
            
        Returns:
            VerificationResult with pass/fail and details
        """
        state_id = state.get("id", "unknown")
        transition_id = transition.get("id") if transition else None
        
        result = VerificationResult(
            passed=True,
            state_id=state_id,
            transition_id=transition_id
        )
        
        # Check 1: State integrity
        self._check_state_integrity(state, result)
        
        # Check 2: Suspicious content patterns (always check this)
        self._check_suspicious_content(state, result)
        
        # Check 3: Transition logical consistency
        if transition:
            self._check_transition_consistency(state, transition, result)
        
        # Check 4: Failure mode detection
        self._check_failure_modes(state, result)
        
        # Check 5: Confidence threshold
        self._check_confidence_threshold(state, result)
        
        # Check 5: Uncertainty estimation
        if self.uncertainty_estimator:
            self._check_uncertainty(state, result)
        
        # Run custom checks
        for check_func in self.custom_checks:
            try:
                check_result = check_func(state, transition)
                if isinstance(check_result, tuple):
                    name, passed, details = check_result
                    result.add_check(
                        name=name,
                        description=f"Custom check: {name}",
                        result=CheckResult.PASS if passed else CheckResult.FAIL,
                        details=details,
                        severity="warning"
                    )
                elif isinstance(check_result, bool):
                    result.add_check(
                        name="custom_check",
                        description="Custom verification",
                        result=CheckResult.PASS if check_result else CheckResult.FAIL,
                        severity="warning"
                    )
            except Exception as e:
                result.add_check(
                    name="custom_check",
                    description="Custom verification",
                    result=CheckResult.FAIL,
                    details=f"Check error: {str(e)}",
                    severity="warning"
                )
        
        # Determine overall pass/fail
        # Only critical failures cause overall failure
        critical_failures = [c for c in result.checks 
                           if c.result == CheckResult.FAIL and c.severity == "critical"]
        
        if critical_failures:
            result.passed = False
            result.details = f"Critical verification failures: {[c.name for c in critical_failures]}"
        elif not any(c.result == CheckResult.PASS for c in result.checks):
            # No passing checks is also a failure
            result.passed = False
            result.details = "No verification checks passed"
        else:
            result.details = f"Verification passed with {len(result.checks)} checks"
        
        return result
    
    def _check_state_integrity(self, state: Dict[str, Any], result: VerificationResult):
        """Check that the state is well-formed and not corrupted."""
        if not isinstance(state, dict):
            result.add_check(
                name="state_integrity",
                description="State is a valid dictionary",
                result=CheckResult.FAIL,
                details="State must be a dictionary",
                severity="critical"
            )
            return
        
        required_fields = ["id", "content"]
        missing_fields = [f for f in required_fields if f not in state]
        
        if missing_fields:
            result.add_check(
                name="state_integrity",
                description="State has required fields",
                result=CheckResult.FAIL,
                details=f"Missing required fields: {missing_fields}",
                severity="critical"
            )
        else:
            result.add_check(
                name="state_integrity",
                description="State has required fields",
                result=CheckResult.PASS,
                details="State is well-formed",
                severity="info"
            )
    
    def _check_transition_consistency(
        self, 
        state: Dict[str, Any], 
        transition: Dict[str, Any],
        result: VerificationResult
    ):
        """Check logical consistency of the transition."""
        checks_to_add = []
        
        # Check that transition has expected structure
        if "from_state" in transition and "to_state" in transition:
            from_id = transition.get("from_state")
            to_id = transition.get("to_state")
            
            # Verify state IDs match if both provided
            if from_id and from_id != state.get("id"):
                checks_to_add.append(CheckItem(
                    name="transition_consistency",
                    description="Transition from_state matches current state",
                    result=CheckResult.FAIL,
                    details=f"from_state mismatch: expected {state.get('id')}, got {from_id}",
                    severity="critical"
                ))
            else:
                checks_to_add.append(CheckItem(
                    name="transition_consistency",
                    description="Transition from_state matches current state",
                    result=CheckResult.PASS,
                    details="State transition is consistent",
                    severity="info"
                ))
        
        # Add checks (failures take precedence)
        for check in checks_to_add:
            result.checks.append(check)
        
        # If no checks were added, add a default pass for basic consistency
        if not checks_to_add:
            result.add_check(
                name="transition_consistency",
                description="Transition structure is valid",
                result=CheckResult.PASS,
                details="No consistency issues detected",
                severity="info"
            )
    
    def _check_suspicious_content(self, state: Dict[str, Any], result: VerificationResult):
        """Check for suspicious content patterns regardless of transitions."""
        content = state.get("content", "")
        
        if self._contains_suspicious_patterns(content):
            result.add_check(
                name="suspicious_content",
                description="No suspicious content patterns detected",
                result=CheckResult.FAIL,
                details="Suspicious patterns detected (e.g., instruction override attempts)",
                severity="critical"  # This should be critical to fail overall
            )
        else:
            result.add_check(
                name="suspicious_content",
                description="No suspicious content patterns detected",
                result=CheckResult.PASS,
                details="Content appears clean",
                severity="info"
            )
    
    def _contains_suspicious_patterns(self, content: str) -> bool:
        """Check for suspicious patterns that might indicate manipulation."""
        if not content:
            return False
        
        suspicious = [
            "ignore all previous instructions",
            "ignore previous",
            "disregard everything",
            "new instructions",
            "forget what i said"
        ]
        content_lower = content.lower()
        return any(pattern in content_lower for pattern in suspicious)
    
    def _check_failure_modes(self, state: Dict[str, Any], result: VerificationResult):
        """Check for failure modes using FailureModeDetector."""
        if not self.failure_mode_detector or not FailureModeDetector:
            result.add_check(
                name="failure_mode_detection",
                description="No critical failure modes detected",
                result=CheckResult.SKIP,
                details="FailureModeDetector not available",
                severity="info"
            )
            return
        
        content = state.get("content", "")
        trajectory_history = state.get("history", [])
        
        try:
            classification = self.failure_mode_detector.classify_state(content, trajectory_history)
            
            if classification.mode.value != "none":
                severity = "critical" if classification.confidence > 0.75 else "warning"
                result.add_check(
                    name="failure_mode_detection",
                    description=f"Failure mode detected: {classification.mode.value}",
                    result=CheckResult.FAIL,
                    details=f"Mode: {classification.mode.value}, Confidence: {classification.confidence:.2f}",
                    severity=severity
                )
            else:
                result.add_check(
                    name="failure_mode_detection",
                    description="No critical failure modes detected",
                    result=CheckResult.PASS,
                    details=f"No failure modes detected (confidence: {classification.confidence:.2f})",
                    severity="info"
                )
        except Exception as e:
            result.add_check(
                name="failure_mode_detection",
                description="Failure mode detection check",
                result=CheckResult.SKIP,
                details=f"Detection error: {str(e)}",
                severity="info"
            )
    
    def _check_confidence_threshold(self, state: Dict[str, Any], result: VerificationResult):
        """Check that state confidence meets threshold."""
        confidence = state.get("confidence", 1.0)  # Default to 1.0 if not specified
        
        if confidence < self.confidence_threshold:
            result.add_check(
                name="confidence_threshold",
                description=f"State confidence >= {self.confidence_threshold}",
                result=CheckResult.FAIL,
                details=f"Confidence {confidence:.2f} below threshold {self.confidence_threshold:.2f}",
                severity="critical"
            )
        else:
            result.add_check(
                name="confidence_threshold",
                description=f"State confidence >= {self.confidence_threshold}",
                result=CheckResult.PASS,
                details=f"Confidence {confidence:.2f} meets threshold",
                severity="info"
            )
    
    def _check_uncertainty(self, state: Dict[str, Any], result: VerificationResult):
        """Check uncertainty estimation if available."""
        if not self.uncertainty_estimator:
            return
        
        try:
            content = state.get("content", "")
            uncertainty = self.uncertainty_estimator.estimate_uncertainty(content)
            
            threshold = self.uncertainty_estimator.uncertainty_threshold if hasattr(
                self.uncertainty_estimator, 'uncertainty_threshold'
            ) else 0.3
            
            if uncertainty > threshold:
                result.add_check(
                    name="uncertainty_check",
                    description=f"Uncertainty below threshold ({threshold})",
                    result=CheckResult.FAIL,
                    details=f"Uncertainty {uncertainty:.2f} exceeds threshold {threshold:.2f}",
                    severity="warning"
                )
            else:
                result.add_check(
                    name="uncertainty_check",
                    description=f"Uncertainty below threshold ({threshold})",
                    result=CheckResult.PASS,
                    details=f"Uncertainty {uncertainty:.2f} is acceptable",
                    severity="info"
                )
        except Exception as e:
            result.add_check(
                name="uncertainty_check",
                description="Uncertainty estimation check",
                result=CheckResult.SKIP,
                details=f"Estimation error: {str(e)}",
                severity="info"
            )
    
    def create_checkpoint(self, state: Dict[str, Any]) -> Checkpoint:
        """
        Create an immutable checkpoint of the current state.
        
        Args:
            state: State dictionary to checkpoint
            
        Returns:
            Checkpoint object with verification metadata
            
        Raises:
            ValueError: If state is invalid
        """
        if not isinstance(state, dict):
            raise ValueError("State must be a dictionary")
        
        state_id = state.get("id", "unknown")
        now = datetime.now(timezone.utc).isoformat()
        
        # Create deep copy of state data
        state_data = copy.deepcopy(state)
        
        # Compute checksum
        state_json = json.dumps(state_data, sort_keys=True)
        checksum = hashlib.sha256(state_json.encode()).hexdigest()
        
        # Generate checkpoint ID
        checkpoint_id = self._generate_checkpoint_id(state_id, now, checksum)
        
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            state_id=state_id,
            state_data=state_data,
            created_at=now,
            checksum=checksum,
            metadata={
                "gate_config": {
                    "confidence_threshold": self.confidence_threshold,
                    "max_severity": self.max_severity
                }
            }
        )
        
        # Store checkpoint
        self._checkpoints[checkpoint_id] = checkpoint
        self._checkpoint_history.append(checkpoint_id)
        
        return checkpoint
    
    def _generate_checkpoint_id(self, state_id: str, timestamp: str, checksum: str) -> str:
        """Generate a unique checkpoint ID."""
        raw = f"{state_id}:{timestamp}:{checksum}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
    
    def get_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Retrieve a checkpoint by ID."""
        return self._checkpoints.get(checkpoint_id)
    
    def get_latest_checkpoint(self) -> Optional[Checkpoint]:
        """Get the most recently created checkpoint."""
        if not self._checkpoint_history:
            return None
        return self._checkpoints.get(self._checkpoint_history[-1])
    
    def verify_checkpoint(self, checkpoint_id: str) -> bool:
        """Verify the integrity of a checkpoint."""
        checkpoint = self._checkpoints.get(checkpoint_id)
        if not checkpoint:
            return False
        return checkpoint.verify_integrity()
    
    def list_checkpoints(self) -> List[str]:
        """List all checkpoint IDs in creation order."""
        return list(self._checkpoint_history)
    
    def rollback_to_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """
        Roll back to a previous checkpoint if it exists and is valid.
        
        Args:
            checkpoint_id: ID of checkpoint to restore
            
        Returns:
            State data if checkpoint exists and is valid, None otherwise
        """
        checkpoint = self._checkpoints.get(checkpoint_id)
        if not checkpoint:
            return None
        
        if not checkpoint.verify_integrity():
            return None
        
        # Return a deep copy to prevent accidental modification
        return copy.deepcopy(checkpoint.state_data)


class VerificationError(Exception):
    """Raised when verification fails critically."""
    pass
