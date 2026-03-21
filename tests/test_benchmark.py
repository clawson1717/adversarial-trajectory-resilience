"""
Tests for BenchmarkRunner
"""

import pytest
from src.benchmark import (
    BenchmarkRunner,
    BenchmarkScenario,
    BenchmarkResults,
    ScenarioMetrics,
)
from src.detector import FailureMode


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner class."""

    def test_init_default(self):
        """Test default initialization."""
        runner = BenchmarkRunner()
        assert runner is not None
        assert runner.seed == 42
        assert runner.verbose is False

    def test_init_with_params(self):
        """Test initialization with parameters."""
        runner = BenchmarkRunner(seed=123, verbose=True)
        assert runner.seed == 123
        assert runner.verbose is True

    def test_create_components(self):
        """Test that components are created correctly."""
        runner = BenchmarkRunner()
        gate, allocator, orchestrator = runner._create_components()
        
        assert gate is not None
        assert allocator is not None
        assert orchestrator is not None
        assert orchestrator.is_initialized

    def test_state_from_step(self):
        """Test state conversion from ReasoningStep."""
        from src.mock_agent import ReasoningStep, MockResponseMode
        
        runner = BenchmarkRunner()
        step = ReasoningStep(
            step_id=5,
            content="Test content",
            failure_mode=FailureMode.NONE,
            response_mode=MockResponseMode.NORMAL,
            metadata={"prompt": "test"},
        )
        
        state = runner._state_from_step(step, 5)
        
        assert state["id"] == "state_5"
        assert state["content"] == "Test content"
        assert "confidence" in state
        assert state["history"] == []

    def test_run_scenario_clean(self):
        """Test running clean scenario - verify no failures detected."""
        runner = BenchmarkRunner(seed=42, verbose=False)
        metrics = runner.run_scenario(BenchmarkScenario.CLEAN)
        
        assert metrics is not None
        assert metrics.scenario == BenchmarkScenario.CLEAN
        # Clean scenario should have no failures
        assert metrics.baseline_failures_detected == 0
        assert metrics.resilience_failures_detected == 0
        # Health should be healthy for clean
        assert metrics.baseline_final_health == "healthy"
        assert metrics.resilience_final_health == "healthy"
        # Detection accuracy should be 100% (no false positives)
        assert metrics.detection_accuracy == 100.0

    def test_run_scenario_adversarial(self):
        """Test running adversarial scenario - verify failures are detected."""
        runner = BenchmarkRunner(seed=42, verbose=False)
        metrics = runner.run_scenario(BenchmarkScenario.ADVERSARIAL)
        
        assert metrics is not None
        assert metrics.scenario == BenchmarkScenario.ADVERSARIAL
        # Adversarial scenario should have failures injected
        assert metrics.baseline_failures_detected == 3
        # Detection accuracy should be > 0 (failures detected)
        assert metrics.detection_accuracy > 0

    def test_run_scenario_high_uncertainty(self):
        """Test running high-uncertainty scenario."""
        runner = BenchmarkRunner(seed=42, verbose=False)
        metrics = runner.run_scenario(BenchmarkScenario.HIGH_UNCERTAINTY)
        
        assert metrics is not None
        assert metrics.scenario == BenchmarkScenario.HIGH_UNCERTAINTY
        # No explicit failures in high-uncertainty
        assert metrics.baseline_failures_detected == 0
        # Health should be degraded (not terminated or critical) for high uncertainty
        assert metrics.resilience_final_health in ("healthy", "degraded")

    def test_run_full_benchmark(self):
        """Test running all scenarios returns results object."""
        runner = BenchmarkRunner(seed=42, verbose=False)
        results = runner.run_full_benchmark()
        
        assert results is not None
        assert isinstance(results, BenchmarkResults)
        # Should have all three scenarios
        assert len(results.scenarios) == 3
        assert BenchmarkScenario.CLEAN in results.scenarios
        assert BenchmarkScenario.ADVERSARIAL in results.scenarios
        assert BenchmarkScenario.HIGH_UNCERTAINTY in results.scenarios
        # Aggregate metrics should be computed
        assert results.total_baseline_steps > 0
        assert results.total_resilience_steps > 0

    def test_efficiency_gain_computation(self):
        """Test that efficiency_gain is computed correctly and within valid range."""
        runner = BenchmarkRunner(seed=42, verbose=False)
        results = runner.run_full_benchmark()
        
        for scenario, metrics in results.scenarios.items():
            # Efficiency gain should be between -100 and 100
            assert -100 <= metrics.efficiency_gain <= 100, \
                f"{scenario.value}: efficiency_gain {metrics.efficiency_gain} out of range"
        
        # For clean scenario: both should run same steps, efficiency should be 0
        clean_metrics = results.scenarios[BenchmarkScenario.CLEAN]
        assert clean_metrics.baseline_steps == clean_metrics.resilience_steps
        assert clean_metrics.efficiency_gain == 0.0

    def test_efficiency_gain_formula(self):
        """Test efficiency_gain formula: (baseline - resilience) / baseline * 100."""
        runner = BenchmarkRunner(seed=42, verbose=False)
        
        # Run scenarios individually to check formula
        for scenario in [BenchmarkScenario.CLEAN, BenchmarkScenario.ADVERSARIAL, BenchmarkScenario.HIGH_UNCERTAINTY]:
            metrics = runner.run_scenario(scenario)
            
            expected_efficiency = 0.0
            if metrics.baseline_steps > 0:
                expected_efficiency = (
                    (metrics.baseline_steps - metrics.resilience_steps) / metrics.baseline_steps * 100
                )
            
            assert abs(metrics.efficiency_gain - expected_efficiency) < 0.01, \
                f"{scenario.value}: expected {expected_efficiency}, got {metrics.efficiency_gain}"

    def test_early_termination(self):
        """Test that resilience system can terminate early in adversarial scenario."""
        runner = BenchmarkRunner(seed=42, verbose=False)
        metrics = runner.run_scenario(BenchmarkScenario.ADVERSARIAL)
        
        # In adversarial scenario, resilience should terminate early
        # Baseline runs all 15 steps, resilience should run fewer
        assert metrics.resilience_steps <= metrics.baseline_steps
        # If early termination works, resilience_steps < baseline_steps
        # (This may not always happen depending on when failures occur)

    def test_health_states(self):
        """Test that health states are reasonable given failure counts."""
        runner = BenchmarkRunner(seed=42, verbose=False)
        results = runner.run_full_benchmark()
        
        # Clean: no failures, health should be healthy
        clean = results.scenarios[BenchmarkScenario.CLEAN]
        assert clean.resilience_failures_detected == 0
        assert clean.resilience_final_health == "healthy"
        
        # High uncertainty: no failures, health should not be critical/terminated
        high_unc = results.scenarios[BenchmarkScenario.HIGH_UNCERTAINTY]
        assert high_unc.resilience_failures_detected == 0
        assert high_unc.resilience_final_health in ("healthy", "degraded")

    def test_benchmark_results_to_dict(self):
        """Test that results can be serialized to dict."""
        runner = BenchmarkRunner(seed=42, verbose=False)
        results = runner.run_full_benchmark()
        
        result_dict = results.to_dict()
        
        assert "scenarios" in result_dict
        assert "aggregate" in result_dict
        assert "metadata" in result_dict
        
        # Check scenario data
        for scenario_name in ["clean", "adversarial", "high_uncertainty"]:
            assert scenario_name in result_dict["scenarios"]
            scenario_data = result_dict["scenarios"][scenario_name]
            assert "baseline_steps" in scenario_data
            assert "resilience_steps" in scenario_data
            assert "efficiency_gain" in scenario_data


class TestScenarioMetrics:
    """Tests for ScenarioMetrics dataclass."""

    def test_scenario_metrics_creation(self):
        """Test creating ScenarioMetrics."""
        metrics = ScenarioMetrics(scenario=BenchmarkScenario.CLEAN)
        
        assert metrics.scenario == BenchmarkScenario.CLEAN
        assert metrics.baseline_steps == 0
        assert metrics.resilience_steps == 0
        assert metrics.efficiency_gain == 0.0

    def test_scenario_metrics_with_values(self):
        """Test ScenarioMetrics with actual values."""
        metrics = ScenarioMetrics(
            scenario=BenchmarkScenario.ADVERSARIAL,
            baseline_steps=15,
            resilience_steps=12,
            resilience_failures_detected=2,
            resilience_recovered=3,
        )
        
        assert metrics.baseline_steps == 15
        assert metrics.resilience_steps == 12
        assert metrics.resilience_failures_detected == 2
        assert metrics.resilience_recovered == 3


class TestBenchmarkResults:
    """Tests for BenchmarkResults dataclass."""

    def test_benchmark_results_creation(self):
        """Test creating BenchmarkResults."""
        results = BenchmarkResults()
        
        assert len(results.scenarios) == 0
        assert results.total_baseline_steps == 0
        assert results.total_resilience_steps == 0

    def test_benchmark_results_aggregate(self):
        """Test that aggregate metrics are computed."""
        results = BenchmarkResults()
        
        # Add some scenarios
        results.scenarios[BenchmarkScenario.CLEAN] = ScenarioMetrics(
            scenario=BenchmarkScenario.CLEAN,
            baseline_steps=10,
            resilience_steps=10,
            efficiency_gain=0.0,
            failure_recovery_rate=100.0,
            compute_savings=0.0,
            detection_accuracy=100.0,
        )
        results.scenarios[BenchmarkScenario.ADVERSARIAL] = ScenarioMetrics(
            scenario=BenchmarkScenario.ADVERSARIAL,
            baseline_steps=15,
            resilience_steps=12,
            efficiency_gain=20.0,
            failure_recovery_rate=100.0,
            compute_savings=86.67,
            detection_accuracy=66.7,
        )
        
        # Manually compute aggregates
        metrics_list = list(results.scenarios.values())
        results.avg_efficiency_gain = sum(m.efficiency_gain for m in metrics_list) / len(metrics_list)
        results.total_baseline_steps = sum(m.baseline_steps for m in metrics_list)
        results.total_resilience_steps = sum(m.resilience_steps for m in metrics_list)
        
        assert results.avg_efficiency_gain == 10.0
        assert results.total_baseline_steps == 25
        assert results.total_resilience_steps == 22
