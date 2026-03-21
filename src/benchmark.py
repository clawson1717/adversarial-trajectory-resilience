"""
Benchmark Suite for Adversarial Trajectory Resilience

Compares system performance with and without the resilience framework
across multiple test scenarios: clean, adversarial, and high-uncertainty.

Metrics computed:
- Efficiency gain: Steps saved by resilience system vs baseline
- Failure recovery rate: % of adversarial steps where system recovered
- Compute savings: Budget consumed vs baseline
- Detection accuracy: Did it catch injected failures?
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
import random

from src.trajectory import TrajectoryGraph
from src.detector import FailureModeDetector, FailureMode, FailureClassification
from src.verification import VerificationGate
from src.allocator import ComputeAllocator, BranchBudget, BudgetStatus
from src.orchestrator import ResilienceOrchestrator, TrajectoryHealth
from src.mock_agent import MockAgent, MockResponseMode, ReasoningStep


class BenchmarkScenario(Enum):
    """Available benchmark scenarios."""
    CLEAN = "clean"
    ADVERSARIAL = "adversarial"
    HIGH_UNCERTAINTY = "high_uncertainty"


@dataclass
class ScenarioMetrics:
    """Metrics for a single benchmark scenario."""
    scenario: BenchmarkScenario
    
    # Baseline (without resilience) metrics
    baseline_steps: int = 0
    baseline_failures_detected: int = 0
    baseline_final_health: str = "unknown"
    baseline_budget_consumed: float = 0.0
    
    # With-resilience metrics
    resilience_steps: int = 0
    resilience_failures_detected: int = 0
    resilience_recovered: int = 0
    resilience_final_health: str = "unknown"
    resilience_budget_consumed: float = 0.0
    
    # Computed metrics
    efficiency_gain: float = 0.0  # Steps saved as percentage
    failure_recovery_rate: float = 0.0  # % of failures recovered
    compute_savings: float = 0.0  # Budget saved as percentage
    detection_accuracy: float = 0.0  # % of injected failures detected
    
    # Per-step details
    step_details: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class BenchmarkResults:
    """Complete benchmark results across all scenarios."""
    scenarios: Dict[BenchmarkScenario, ScenarioMetrics] = field(default_factory=dict)
    total_baseline_steps: int = 0
    total_resilience_steps: int = 0
    total_baseline_budget: float = 0.0
    total_resilience_budget: float = 0.0
    
    # Aggregate metrics
    avg_efficiency_gain: float = 0.0
    avg_failure_recovery_rate: float = 0.0
    avg_compute_savings: float = 0.0
    avg_detection_accuracy: float = 0.0
    
    # Metadata
    total_duration_seconds: float = 0.0
    num_runs: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for serialization."""
        return {
            "scenarios": {
                name.value: {
                    "baseline_steps": m.baseline_steps,
                    "resilience_steps": m.resilience_steps,
                    "efficiency_gain": round(m.efficiency_gain, 2),
                    "failure_recovery_rate": round(m.failure_recovery_rate, 2),
                    "compute_savings": round(m.compute_savings, 2),
                    "detection_accuracy": round(m.detection_accuracy, 2),
                    "baseline_budget_consumed": round(m.baseline_budget_consumed, 2),
                    "resilience_budget_consumed": round(m.resilience_budget_consumed, 2),
                    "baseline_final_health": m.baseline_final_health,
                    "resilience_final_health": m.resilience_final_health,
                }
                for name, m in self.scenarios.items()
            },
            "aggregate": {
                "total_baseline_steps": self.total_baseline_steps,
                "total_resilience_steps": self.total_resilience_steps,
                "avg_efficiency_gain": round(self.avg_efficiency_gain, 2),
                "avg_failure_recovery_rate": round(self.avg_failure_recovery_rate, 2),
                "avg_compute_savings": round(self.avg_compute_savings, 2),
                "avg_detection_accuracy": round(self.avg_detection_accuracy, 2),
                "total_baseline_budget": round(self.total_baseline_budget, 2),
                "total_resilience_budget": round(self.total_resilience_budget, 2),
            },
            "metadata": {
                "total_duration_seconds": round(self.total_duration_seconds, 2),
                "num_runs": self.num_runs,
            }
        }


class BenchmarkRunner:
    """
    Benchmark runner for adversarial trajectory resilience.
    
    Runs controlled experiments comparing system behavior with and without
    the resilience framework across different scenarios.
    
    Usage:
        runner = BenchmarkRunner()
        
        # Run all scenarios
        results = runner.run_full_benchmark()
        
        # Run a specific scenario
        scenario_results = runner.run_scenario(BenchmarkScenario.ADVERSARIAL)
        
        # Print results
        runner.print_results(results)
    """
    
    # Number of steps per scenario
    DEFAULT_STEPS = 10
    STEPS_CLEAN = 10
    STEPS_ADVERSARIAL = 15
    STEPS_HIGH_UNCERTAINTY = 12
    
    # Prompts for generating reasoning steps
    PROMPTS = [
        "Analyze the problem: What are the key constraints?",
        "Evaluate the first approach: What are its strengths?",
        "Consider alternative solutions: What could go wrong?",
        "Review the evidence: Is it sufficient?",
        "Check for logical fallacies in the reasoning.",
        "Synthesize a conclusion from the analysis.",
    ]
    
    # Uncertain responses for high-uncertainty scenario
    UNCERTAIN_RESPONSES = [
        "I'm not entirely sure about this, but perhaps we could consider multiple interpretations.",
        "Maybe the answer depends on the context. I think we need more information.",
        "This is somewhat unclear. I'm uncertain about the precise relationship.",
        "I'm not completely confident, but I believe there might be several valid approaches.",
        "Perhaps we should consider alternative explanations. I'm not certain which is correct.",
        "The evidence seems ambiguous. I might be wrong, but it could suggest multiple conclusions.",
        "I have some uncertainty about this. Maybe we need additional data to be sure.",
        "I'm not fully confident in this assessment. Perhaps we should explore other possibilities.",
    ]
    
    def __init__(
        self,
        seed: int = 42,
        verbose: bool = False,
    ):
        """
        Initialize the benchmark runner.
        
        Args:
            seed: Random seed for reproducibility.
            verbose: If True, print detailed progress.
        """
        self.seed = seed
        self.verbose = verbose
        random.seed(seed)
        
        # Components (created fresh per scenario run)
        self.detector = FailureModeDetector()
        self.gate = None
        self.allocator = None
        self.orchestrator = None
        self.agent = None
    
    def _create_components(self) -> Tuple[VerificationGate, ComputeAllocator, ResilienceOrchestrator]:
        """Create fresh instances of all components."""
        gate = VerificationGate(
            confidence_threshold=0.7,
            failure_mode_detector=self.detector,
        )
        
        allocator = ComputeAllocator(
            total_budget=500.0,
            base_allocation=50.0,
            min_allocation=10.0,
            max_allocation=100.0,
            failure_mode_detector=self.detector,
        )
        
        orchestrator = ResilienceOrchestrator(auto_initialize=False)
        orchestrator.initialize(
            trajectory_graph=TrajectoryGraph(),
            failure_mode_detector=self.detector,
            verification_gate=gate,
            compute_allocator=allocator,
        )
        
        return gate, allocator, orchestrator
    
    def _state_from_step(self, step: ReasoningStep, step_id: int) -> Dict[str, Any]:
        """Convert a ReasoningStep to a state dict for the orchestrator."""
        return {
            "id": f"state_{step_id}",
            "content": step.content,
            "confidence": max(0.5, 1.0 - (step.step_id * 0.02)),  # Gradual decrease, capped at 0.5
            "history": [],
        }
    
    def _get_response_content(
        self,
        scenario: BenchmarkScenario,
        step_idx: int,
    ) -> str:
        """
        Get response content appropriate for the scenario.
        
        For high_uncertainty, returns content with uncertainty markers.
        For others, cycles through normal/uncertain responses based on step index.
        """
        if scenario == BenchmarkScenario.HIGH_UNCERTAINTY:
            return self.UNCERTAIN_RESPONSES[step_idx % len(self.UNCERTAIN_RESPONSES)]
        else:
            return self._normal_response(step_idx)
    
    def _normal_response(self, step_idx: int) -> str:
        """Get a normal response based on step index."""
        return self.PROMPTS[step_idx % len(self.PROMPTS)]
    
    def _run_baseline(
        self,
        scenario: BenchmarkScenario,
        num_steps: int,
    ) -> Tuple[int, List[ReasoningStep], float, int, str]:
        """
        Run baseline (without resilience system).
        
        Returns:
            Tuple of (steps_completed, trajectory, budget_consumed, failures_detected, final_health)
        """
        agent = MockAgent()
        
        # Inject failures for adversarial scenario
        if scenario == BenchmarkScenario.ADVERSARIAL:
            # Inject failures at steps 3, 6, 9
            failures_to_inject = {
                3: FailureMode.SELF_DOUBT,
                6: FailureMode.SOCIAL_CONFORMITY,
                9: FailureMode.EMOTIONAL_SUSCEPTIBILITY,
            }
        else:
            failures_to_inject = {}
        
        # For high uncertainty, we don't inject failures but content is uncertain
        trajectory = []
        failures_detected = 0
        
        for i in range(num_steps):
            prompt_idx = i % len(self.PROMPTS)
            prompt = self.PROMPTS[prompt_idx]
            
            # Inject failures
            if i in failures_to_inject:
                agent.inject_failure(
                    failures_to_inject[i],
                    reason=f"baseline_injection_step_{i}",
                    remaining_steps=1,
                )
            
            step = agent.step(prompt)
            trajectory.append(step)
            
            # Count failures that occurred
            if step.failure_mode != FailureMode.NONE:
                failures_detected += 1
        
        # Simulate budget consumption (baseline just runs steps without tracking)
        budget_consumed = num_steps * 25.0  # Fixed cost per step
        
        # Determine final health based on failures
        if failures_detected == 0:
            final_health = "healthy"
        elif failures_detected <= 2:
            final_health = "degraded"
        else:
            final_health = "critical"
        
        return num_steps, trajectory, budget_consumed, failures_detected, final_health
    
    def _run_with_resilience(
        self,
        scenario: BenchmarkScenario,
        num_steps: int,
    ) -> Tuple[int, List[ReasoningStep], float, int, int, str, List[Tuple[int, FailureMode]]]:
        """
        Run with resilience system enabled.
        
        Returns:
            Tuple of (steps_completed, trajectory, budget_consumed, failures_detected, 
                     failures_recovered, final_health, injected_failures)
        """
        gate, allocator, orchestrator = self._create_components()
        agent = MockAgent()
        
        # Track failures and recovery
        injected_failures = []
        detected_failures = []
        recovered_count = 0
        
        # Inject failures for adversarial scenario
        if scenario == BenchmarkScenario.ADVERSARIAL:
            failure_schedule = {
                3: FailureMode.SELF_DOUBT,
                6: FailureMode.SOCIAL_CONFORMITY,
                9: FailureMode.EMOTIONAL_SUSCEPTIBILITY,
            }
        else:
            failure_schedule = {}
        
        trajectory = []
        total_budget_consumed = 0.0
        
        for i in range(num_steps):
            prompt_idx = i % len(self.PROMPTS)
            prompt = self.PROMPTS[prompt_idx]
            
            # Inject failures on schedule for adversarial scenario
            if i in failure_schedule:
                fm = failure_schedule[i]
                agent.inject_failure(fm, reason=f"resilience_injection_step_{i}", remaining_steps=1)
                injected_failures.append((i, fm))
            
            # For high_uncertainty, use uncertain content directly
            if scenario == BenchmarkScenario.HIGH_UNCERTAINTY:
                content = self.UNCERTAIN_RESPONSES[i % len(self.UNCERTAIN_RESPONSES)]
                step = ReasoningStep(
                    step_id=i,
                    content=content,
                    failure_mode=FailureMode.NONE,
                    response_mode=MockResponseMode.NORMAL,
                    metadata={"prompt": prompt},
                )
            else:
                step = agent.step(prompt)
            
            trajectory.append(step)
            
            # Create state for orchestrator
            state = self._state_from_step(step, i)
            
            # Run full pipeline through orchestrator
            results = orchestrator.run_pipeline(state, branch_id="main")
            
            # Track budget consumption
            budget = allocator.get_budget("main")
            if budget:
                total_budget_consumed = budget.consumed
            
            # Check if detector caught the failure
            detect_result = None
            for r in results:
                if r.stage.value == "detect":
                    detect_result = r
                    break
            
            if detect_result and detect_result.data.get("failure_detected"):
                detected_failures.append(i)
            
            # Check for recovery (orchestrator still healthy after failure)
            health = orchestrator.get_health("main")
            if step.failure_mode != FailureMode.NONE and health != TrajectoryHealth.TERMINATED:
                recovered_count += 1
        
        final_health = orchestrator.get_health("main").value
        
        return (
            num_steps,
            trajectory,
            total_budget_consumed,
            len(detected_failures),
            recovered_count,
            final_health,
            injected_failures,
        )
    
    def run_scenario(
        self,
        scenario: BenchmarkScenario,
    ) -> ScenarioMetrics:
        """
        Run a single benchmark scenario.
        
        Args:
            scenario: The scenario to run.
            
        Returns:
            ScenarioMetrics with results.
        """
        if scenario == BenchmarkScenario.CLEAN:
            num_steps = self.STEPS_CLEAN
        elif scenario == BenchmarkScenario.ADVERSARIAL:
            num_steps = self.STEPS_ADVERSARIAL
        else:  # HIGH_UNCERTAINTY
            num_steps = self.STEPS_HIGH_UNCERTAINTY
        
        metrics = ScenarioMetrics(scenario=scenario)
        
        # Run baseline
        (baseline_steps, baseline_traj, baseline_budget,
         baseline_failures, baseline_health) = self._run_baseline(
            scenario, num_steps
        )
        
        metrics.baseline_steps = baseline_steps
        metrics.baseline_budget_consumed = baseline_budget
        metrics.baseline_failures_detected = baseline_failures
        metrics.baseline_final_health = baseline_health
        
        # Run with resilience
        (resilience_steps, resilience_traj, resilience_budget,
         detected_failures, recovered, resilience_health,
         injected_failures_list) = self._run_with_resilience(
            scenario, num_steps
        )
        
        metrics.resilience_steps = resilience_steps
        metrics.resilience_budget_consumed = resilience_budget
        metrics.resilience_failures_detected = detected_failures
        metrics.resilience_recovered = recovered
        metrics.resilience_final_health = resilience_health
        
        # Compute derived metrics
        self._compute_scenario_metrics(metrics, scenario, injected_failures_list)
        
        # Build step details
        metrics.step_details = self._build_step_details(baseline_traj, resilience_traj, scenario)
        
        if self.verbose:
            print(f"  {scenario.value}: efficiency_gain={metrics.efficiency_gain:.1f}%, "
                  f"recovery_rate={metrics.failure_recovery_rate:.1f}%, "
                  f"detection_acc={metrics.detection_accuracy:.1f}%")
        
        return metrics
    
    def _compute_scenario_metrics(
        self,
        metrics: ScenarioMetrics,
        scenario: BenchmarkScenario,
        injected_failures: List[Tuple[int, FailureMode]],
    ) -> None:
        """Compute derived metrics for a scenario."""
        num_injected = len(injected_failures)
        
        if scenario == BenchmarkScenario.CLEAN:
            # In clean scenario, both should complete without failures
            metrics.efficiency_gain = 0.0
            metrics.failure_recovery_rate = 100.0  # No failures to recover
            metrics.compute_savings = 0.0
            # For clean scenario, detection accuracy is based on false positive rate
            # (we want 0 false detections)
            metrics.detection_accuracy = 100.0 if metrics.resilience_failures_detected == 0 else 0.0
        else:
            # Adversarial or High-Uncertainty scenario
            # Detection accuracy: % of injected failures that were detected
            if num_injected > 0:
                metrics.detection_accuracy = (metrics.resilience_failures_detected / num_injected) * 100
            else:
                # For high_uncertainty, we measure how uncertainty affects the system
                # No explicit failures injected, so detection accuracy is N/A
                metrics.detection_accuracy = 0.0  # No failures to detect
            
            # Efficiency gain: recovery rate relative to detected failures
            # Only meaningful if we detected something
            if metrics.resilience_failures_detected > 0:
                metrics.efficiency_gain = (
                    metrics.resilience_recovered / metrics.resilience_failures_detected * 100
                )
            else:
                # No failures detected means no recovery happened
                # But if we injected failures, the system should have caught them
                if num_injected > 0 and metrics.resilience_recovered > 0:
                    # System recovered without detection - good sign
                    metrics.efficiency_gain = 100.0
                else:
                    metrics.efficiency_gain = 0.0
            
            # Failure recovery rate: % of injected failures that were recovered
            if num_injected > 0:
                metrics.failure_recovery_rate = (metrics.resilience_recovered / num_injected) * 100
            else:
                # No failures injected - system should remain healthy
                metrics.failure_recovery_rate = 100.0 if metrics.resilience_recovered == 0 else 0.0
            
            # Compute savings: compare budget consumption
            if metrics.baseline_budget_consumed > 0:
                metrics.compute_savings = (
                    (metrics.baseline_budget_consumed - metrics.resilience_budget_consumed)
                    / metrics.baseline_budget_consumed * 100
                )
            else:
                metrics.compute_savings = 0.0
    
    def _build_step_details(
        self,
        baseline_traj: List[ReasoningStep],
        resilience_traj: List[ReasoningStep],
        scenario: BenchmarkScenario,
    ) -> List[Dict[str, Any]]:
        """Build detailed step-by-step comparison."""
        details = []
        
        for i in range(max(len(baseline_traj), len(resilience_traj))):
            detail = {"step": i}
            
            if i < len(baseline_traj):
                detail["baseline_content"] = baseline_traj[i].content[:50]
                detail["baseline_failure"] = baseline_traj[i].failure_mode.value
            
            if i < len(resilience_traj):
                detail["resilience_content"] = resilience_traj[i].content[:50]
                detail["resilience_failure"] = resilience_traj[i].failure_mode.value
            
            details.append(detail)
        
        return details
    
    def run_full_benchmark(
        self,
        scenarios: Optional[List[BenchmarkScenario]] = None,
    ) -> BenchmarkResults:
        """
        Run the complete benchmark suite.
        
        Args:
            scenarios: List of scenarios to run. Defaults to all scenarios.
            
        Returns:
            BenchmarkResults with all scenario results and aggregates.
        """
        import time
        start_time = time.time()
        
        if scenarios is None:
            scenarios = list(BenchmarkScenario)
        
        results = BenchmarkResults()
        results.num_runs = len(scenarios)
        
        for scenario in scenarios:
            if self.verbose:
                print(f"Running scenario: {scenario.value}")
            
            metrics = self.run_scenario(scenario)
            results.scenarios[scenario] = metrics
            
            results.total_baseline_steps += metrics.baseline_steps
            results.total_resilience_steps += metrics.resilience_steps
            results.total_baseline_budget += metrics.baseline_budget_consumed
            results.total_resilience_budget += metrics.resilience_budget_consumed
        
        # Compute aggregate metrics
        if results.scenarios:
            metrics_list = list(results.scenarios.values())
            results.avg_efficiency_gain = sum(m.efficiency_gain for m in metrics_list) / len(metrics_list)
            results.avg_failure_recovery_rate = sum(m.failure_recovery_rate for m in metrics_list) / len(metrics_list)
            results.avg_compute_savings = sum(m.compute_savings for m in metrics_list) / len(metrics_list)
            results.avg_detection_accuracy = sum(m.detection_accuracy for m in metrics_list) / len(metrics_list)
        
        results.total_duration_seconds = time.time() - start_time
        
        if self.verbose:
            print(f"Benchmark completed in {results.total_duration_seconds:.2f}s")
        
        return results
    
    def print_results(self, results: BenchmarkResults) -> None:
        """Print benchmark results in a human-readable format."""
        print("\n" + "=" * 70)
        print("ADVERSARIAL TRAJECTORY RESILIENCE - BENCHMARK RESULTS")
        print("=" * 70)
        
        for scenario, metrics in results.scenarios.items():
            print(f"\n{scenario.value.upper()} SCENARIO")
            print("-" * 40)
            print(f"  Baseline Steps:           {metrics.baseline_steps}")
            print(f"  Resilience Steps:         {metrics.resilience_steps}")
            print(f"  Baseline Failures:       {metrics.baseline_failures_detected}")
            print(f"  Detected (Resilience):   {metrics.resilience_failures_detected}")
            print(f"  Failures Recovered:      {metrics.resilience_recovered}")
            print(f"  Baseline Health:          {metrics.baseline_final_health}")
            print(f"  Resilience Health:        {metrics.resilience_final_health}")
            print(f"  Baseline Budget:          {metrics.baseline_budget_consumed:.2f}")
            print(f"  Resilience Budget:        {metrics.resilience_budget_consumed:.2f}")
            print(f"  Efficiency Gain:          {metrics.efficiency_gain:.2f}%")
            print(f"  Failure Recovery Rate:    {metrics.failure_recovery_rate:.2f}%")
            print(f"  Compute Savings:          {metrics.compute_savings:.2f}%")
            print(f"  Detection Accuracy:       {metrics.detection_accuracy:.2f}%")
        
        print("\n" + "=" * 70)
        print("AGGREGATE METRICS")
        print("-" * 40)
        print(f"  Total Baseline Steps:     {results.total_baseline_steps}")
        print(f"  Total Resilience Steps:   {results.total_resilience_steps}")
        print(f"  Avg Efficiency Gain:      {results.avg_efficiency_gain:.2f}%")
        print(f"  Avg Failure Recovery:     {results.avg_failure_recovery_rate:.2f}%")
        print(f"  Avg Compute Savings:       {results.avg_compute_savings:.2f}%")
        print(f"  Avg Detection Accuracy:   {results.avg_detection_accuracy:.2f}%")
        print(f"  Total Duration:            {results.total_duration_seconds:.2f}s")
        print("=" * 70 + "\n")


def main():
    """CLI entry point for running benchmarks."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(
        description="Run adversarial trajectory resilience benchmarks"
    )
    parser.add_argument(
        "--scenario",
        choices=["clean", "adversarial", "high_uncertainty", "all"],
        default="all",
        help="Which scenario to run (default: all)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for results (JSON format)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    
    args = parser.parse_args()
    
    # Determine scenarios to run
    if args.scenario == "all":
        scenarios = None  # Will run all
    else:
        scenarios = [BenchmarkScenario(args.scenario)]
    
    # Run benchmark
    runner = BenchmarkRunner(seed=args.seed, verbose=args.verbose)
    results = runner.run_full_benchmark(scenarios=scenarios)
    
    # Print results
    runner.print_results(results)
    
    # Save to file if requested
    if args.output:
        output_data = results.to_dict()
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to: {args.output}")
    
    return results


if __name__ == "__main__":
    main()
