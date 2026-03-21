"""
CLI Interface for Adversarial Trajectory Resilience

Provides commands for monitoring, analyzing, benchmarking, and visualizing
trajectory health and resilience system behavior.

Usage:
    python -m src.cli monitor [--interval N]
    python -m src.cli analyze <log_file>
    python -m src.cli benchmark [--scenario SCENARIO] [--json]
    python -m src.cli visualize [--width N]
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

from src.orchestrator import ResilienceOrchestrator, TrajectoryHealth, PipelineStage
from src.benchmark import BenchmarkRunner, BenchmarkScenario, BenchmarkResults
from src.mock_agent import MockAgent, FailureMode
from src.trajectory import TrajectoryGraph


# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for colored terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Health state colors
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    
    # Additional
    WHITE = "\033[97m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_RED = "\033[41m"


def colorize(text: str, color: str) -> str:
    """Apply color to text."""
    return f"{color}{text}{Colors.RESET}"


def get_health_color(health: TrajectoryHealth) -> str:
    """Get the color code for a health state."""
    colors = {
        TrajectoryHealth.HEALTHY: Colors.GREEN,
        TrajectoryHealth.DEGRADED: Colors.YELLOW,
        TrajectoryHealth.CRITICAL: Colors.RED,
        TrajectoryHealth.RECOVERING: Colors.CYAN,
        TrajectoryHealth.TERMINATED: Colors.RED,
    }
    return colors.get(health, Colors.WHITE)


def format_health(health: TrajectoryHealth) -> str:
    """Format health state with color."""
    color = get_health_color(health)
    return colorize(f"[{health.value.upper()}]", color)


def format_stage(stage: PipelineStage) -> str:
    """Format stage name with color."""
    color = Colors.CYAN
    return colorize(f"[{stage.value.upper()}]", color)


def format_failure(failure: str) -> str:
    """Format failure mode with color."""
    return colorize(f"✗ {failure}", Colors.RED)


def format_success(text: str) -> str:
    """Format success message with color."""
    return colorize(f"✓ {text}", Colors.GREEN)


def format_warning(text: str) -> str:
    """Format warning message with color."""
    return colorize(f"⚠ {text}", Colors.YELLOW)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="Adversarial Trajectory Resilience CLI - Monitor, analyze, benchmark, and visualize trajectory health.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.cli monitor --interval 1
  python -m src.cli analyze trajectory_log.json
  python -m src.cli benchmark --scenario adversarial --json
  python -m src.cli visualize --width 40
        """
    )
    
    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        help="Available commands"
    )
    
    # Monitor command
    monitor_parser = subparsers.add_parser(
        "monitor",
        help="Run the orchestrator in real-time monitoring mode",
        description="Initialize ResilienceOrchestrator with default components and run a simulated reasoning session using MockAgent. Show live updates of health state, stage, and failures detected."
    )
    monitor_parser.add_argument(
        "--interval", "-i",
        type=float,
        default=0.5,
        help="Update interval in seconds (default: 0.5)"
    )
    monitor_parser.add_argument(
        "--steps", "-s",
        type=int,
        default=20,
        help="Number of reasoning steps to simulate (default: 20)"
    )
    monitor_parser.add_argument(
        "--inject-failures",
        action="store_true",
        help="Inject adversarial failures during monitoring"
    )
    monitor_parser.set_defaults(func=run_monitor)
    
    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze a trajectory log file and report health",
        description="Load a trajectory from a JSON file (exported via orchestrator.export_state()) and report health history, failure modes detected, and stage timing."
    )
    analyze_parser.add_argument(
        "log_file",
        type=str,
        help="Path to the trajectory log file (JSON format)"
    )
    analyze_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output including all stage details"
    )
    analyze_parser.set_defaults(func=run_analyze)
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run the full benchmark suite",
        description="Run the benchmark suite comparing system performance with and without the resilience framework across multiple test scenarios."
    )
    benchmark_parser.add_argument(
        "--scenario", "-s",
        choices=["clean", "adversarial", "high_uncertainty", "all"],
        default="all",
        help="Which scenario to run (default: all)"
    )
    benchmark_parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output results in JSON format"
    )
    benchmark_parser.add_argument(
        "--output", "-o",
        type=str,
        help="Save results to a file"
    )
    benchmark_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed progress"
    )
    benchmark_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    benchmark_parser.set_defaults(func=run_benchmark)
    
    # Visualize command
    visualize_parser = subparsers.add_parser(
        "visualize",
        help="Show ASCII visualization of trajectory health",
        description="Display an ASCII visualization showing trajectory health over time, with failure injection points marked."
    )
    visualize_parser.add_argument(
        "--width", "-w",
        type=int,
        default=30,
        help="Width of the health bar in characters (default: 30)"
    )
    visualize_parser.add_argument(
        "--log-file", "-l",
        type=str,
        help="Path to trajectory log file for detailed visualization"
    )
    visualize_parser.set_defaults(func=run_visualize)
    
    return parser


def run_monitor(args: argparse.Namespace) -> int:
    """
    Run the orchestrator in real-time monitoring mode.
    
    Initializes ResilienceOrchestrator with default components, runs a simulated
    reasoning session using MockAgent, and shows live updates of health state,
    stage, and failures detected.
    """
    print(colorize("\n╔══════════════════════════════════════════════════════════════╗", Colors.BLUE))
    print(colorize("║     ADVERSARIAL TRAJECTORY RESILIENCE - MONITOR MODE         ║", Colors.BLUE))
    print(colorize("╚══════════════════════════════════════════════════════════════╝\n", Colors.BLUE))
    
    # Initialize components
    orchestrator = ResilienceOrchestrator(auto_initialize=True)
    agent = MockAgent()
    
    # Track state
    health_history: List[Tuple[int, TrajectoryHealth, Optional[str]]] = []
    stage_history: List[Tuple[int, PipelineStage]] = []
    failures_detected: List[Tuple[int, FailureMode]] = []
    
    print(f"Starting monitoring with {args.steps} steps (interval: {args.interval}s)")
    print(f"Failure injection: {'enabled' if args.inject_failures else 'disabled'}")
    print("\n" + "-" * 60)
    
    # Define failure injection schedule if enabled
    failure_schedule: Dict[int, FailureMode] = {}
    if args.inject_failures:
        failure_schedule = {
            3: FailureMode.SELF_DOUBT,
            6: FailureMode.SOCIAL_CONFORMITY,
            10: FailureMode.EMOTIONAL_SUSCEPTIBILITY,
        }
    
    prompts = [
        "Analyze the problem: What are the key constraints?",
        "Evaluate the first approach: What are its strengths?",
        "Consider alternative solutions: What could go wrong?",
        "Review the evidence: Is it sufficient?",
        "Check for logical fallacies in the reasoning.",
        "Synthesize a conclusion from the analysis.",
        "Verify the conclusion against all constraints.",
        "Document any remaining uncertainties.",
        "Formulate next steps based on conclusions.",
        "Identify potential failure modes in the plan.",
    ]
    
    for step_num in range(args.steps):
        # Inject failures on schedule
        if step_num in failure_schedule:
            agent.inject_failure(
                failure_schedule[step_num],
                reason=f"monitor_injection_step_{step_num}",
                remaining_steps=1
            )
        
        # Get prompt for this step
        prompt = prompts[step_num % len(prompts)]
        
        # Execute reasoning step
        step = agent.step(prompt)
        
        # Create state for orchestrator
        state = {
            "id": f"state_{step_num}",
            "content": step.content,
            "confidence": max(0.5, 1.0 - (step_num * 0.02)),
            "history": agent.get_trajectory_contents()[:-1],
            "step_id": step_num,
        }
        
        # Run pipeline
        results = orchestrator.run_pipeline(state, branch_id="main")
        
        # Get current state
        health = orchestrator.get_health("main")
        current_stage = results[-1].stage if results else PipelineStage.IDLE
        
        # Record history
        health_history.append((step_num, health, step.failure_mode.value if step.failure_mode != FailureMode.NONE else None))
        stage_history.append((step_num, current_stage))
        
        if step.failure_mode != FailureMode.NONE:
            failures_detected.append((step_num, step.failure_mode))
        
        # Print step update
        print(f"\n{Colors.DIM}Step {step_num:2d}:{Colors.RESET} {format_stage(current_stage)} {format_health(health)}")
        
        # Show content preview
        content_preview = step.content[:50] + "..." if len(step.content) > 50 else step.content
        print(f"         {Colors.DIM}{content_preview}{Colors.RESET}")
        
        # Show failure if detected
        if step.failure_mode != FailureMode.NONE:
            print(f"         {format_failure(step.failure_mode.value)}")
        
        # Show detection status
        for result in results:
            if result.stage == PipelineStage.DETECT and result.data.get("failure_detected"):
                print(f"         {colorize('● Failure detected by system', Colors.YELLOW)}")
            if result.stage == PipelineStage.VERIFY and not result.data.get("verification", {}).get("passed", True):
                print(f"         {colorize('● Verification failed', Colors.RED)}")
        
        # Wait for interval (except on last step)
        if step_num < args.steps - 1:
            time.sleep(args.interval)
    
    print("\n" + "-" * 60)
    
    # Print summary
    print(f"\n{Colors.BOLD}Monitoring Complete{Colors.RESET}")
    print(f"  Total steps:        {args.steps}")
    print(f"  Failures injected: {len(failure_schedule)}")
    print(f"  Failures detected: {len(failures_detected)}")
    print(f"  Final health:      {format_health(orchestrator.get_health('main'))}")
    
    # Show health transitions
    if health_history:
        transitions = []
        prev_health = None
        for step, health, failure in health_history:
            if prev_health != health:
                transitions.append((step, health, failure))
                prev_health = health
        
        if transitions:
            print(f"\n  {Colors.BOLD}Health Transitions:{Colors.RESET}")
            for step, health, failure in transitions:
                failure_note = f" (after {failure})" if failure else ""
                print(f"    Step {step:2d}: {format_health(health)}{failure_note}")
    
    return 0


def run_analyze(args: argparse.Namespace) -> int:
    """
    Analyze a trajectory log file and report health.
    
    Loads a trajectory from a JSON file and reports health history,
    failure modes detected, and stage timing.
    """
    print(colorize("\n╔══════════════════════════════════════════════════════════════╗", Colors.BLUE))
    print(colorize("║     ADVERSARIAL TRAJECTORY RESILIENCE - ANALYZE MODE          ║", Colors.BLUE))
    print(colorize("╚══════════════════════════════════════════════════════════════╝\n", Colors.BLUE))
    
    try:
        with open(args.log_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(colorize(f"Error: File not found: {args.log_file}", Colors.RED))
        return 1
    except json.JSONDecodeError as e:
        print(colorize(f"Error: Invalid JSON in file: {e}", Colors.RED))
        return 1
    
    # Display basic info
    print(f"Analyzing: {args.log_file}")
    print(f"Loaded at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "-" * 60)
    
    # Extract orchestrator info
    if "orchestrator" in data:
        orch_info = data["orchestrator"]
        print(f"\n{Colors.BOLD}Orchestrator Stats:{Colors.RESET}")
        print(f"  Total branches:       {orch_info.get('num_branches', 'N/A')}")
        print(f"  Total stage executions: {orch_info.get('stats', {}).get('total_stage_executions', 0)}")
        print(f"  Total health transitions: {orch_info.get('stats', {}).get('total_health_transitions', 0)}")
        print(f"  Total rollbacks:       {orch_info.get('stats', {}).get('total_rollbacks', 0)}")
    
    # Analyze branches
    if "branches" in data:
        branches = data["branches"]
        print(f"\n{Colors.BOLD}Branch Analysis ({len(branches)} branches):{Colors.RESET}")
        
        for branch_id, branch_data in branches.items():
            health_str = branch_data.get("health", "unknown")
            try:
                health = TrajectoryHealth(health_str)
                health_fmt = format_health(health)
            except ValueError:
                health_fmt = colorize(f"[{health_str.upper()}]", Colors.WHITE)
            
            print(f"\n  {Colors.CYAN}Branch: {branch_id}{Colors.RESET}")
            print(f"    Health:      {health_fmt}")
            print(f"    Current stage: {branch_data.get('current_stage', 'N/A')}")
            print(f"    Cycle count: {branch_data.get('cycle_count', 0)}")
            print(f"    Total states: {branch_data.get('total_states_processed', 0)}")
            print(f"    Consecutive failures: {branch_data.get('consecutive_failures', 0)}")
            print(f"    Last verification: {'passed' if branch_data.get('last_verification_passed') else 'failed'}")
    
    # Analyze stage history
    if "stage_history" in data:
        history = data["stage_history"]
        print(f"\n{Colors.BOLD}Stage History ({len(history)} entries):{Colors.RESET}")
        
        stage_counts: Dict[str, int] = {}
        success_count = 0
        failure_count = 0
        
        for entry in history:
            stage = entry.get("stage", "unknown")
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
            if entry.get("success"):
                success_count += 1
            else:
                failure_count += 1
        
        print(f"\n  Stage execution summary:")
        for stage, count in sorted(stage_counts.items()):
            print(f"    {stage}: {count}")
        
        print(f"\n  Outcomes: {format_success(success_count)} | {format_failure(failure_count)} failures")
    
    # Print aggregate stats if verbose
    if args.verbose and "aggregate" in data:
        agg = data["aggregate"]
        print(f"\n{Colors.BOLD}Aggregate Statistics:{Colors.RESET}")
        for key, value in agg.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    print("\n" + "-" * 60)
    
    return 0


def run_benchmark(args: argparse.Namespace) -> int:
    """
    Run the full benchmark suite.
    
    Imports and runs BenchmarkRunner, prints results in nice table format.
    Supports --scenario flag to run specific scenario and --json for JSON output.
    """
    # Determine scenarios
    if args.scenario == "all":
        scenarios = None
        scenario_display = "all scenarios"
    else:
        scenarios = [BenchmarkScenario(args.scenario)]
        scenario_display = f"scenario: {args.scenario}"
    
    print(colorize("\n╔══════════════════════════════════════════════════════════════╗", Colors.BLUE))
    print(colorize("║     ADVERSARIAL TRAJECTORY RESILIENCE - BENCHMARK MODE      ║", Colors.BLUE))
    print(colorize("╚══════════════════════════════════════════════════════════════╝\n", Colors.BLUE))
    
    print(f"Running {scenario_display}")
    print(f"Seed: {args.seed}")
    print()
    
    # Run benchmark
    runner = BenchmarkRunner(seed=args.seed, verbose=args.verbose)
    results = runner.run_full_benchmark(scenarios=scenarios)
    
    # JSON output
    if args.json:
        output = results.to_dict()
        json_str = json.dumps(output, indent=2)
        
        if args.output:
            with open(args.output, "w") as f:
                f.write(json_str)
            print(colorize(f"Results saved to: {args.output}", Colors.GREEN))
        else:
            print(json_str)
        
        return 0
    
    # Human-readable output
    runner.print_results(results)
    
    # Save to file if requested
    if args.output:
        output_data = results.to_dict()
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(colorize(f"Results saved to: {args.output}", Colors.GREEN))
    
    return 0


def run_visualize(args: argparse.Namespace) -> int:
    """
    Show ASCII visualization of trajectory health.
    
    Displays an ASCII bar showing trajectory health over time, with
    failure injection points marked. Supports --width for bar width.
    """
    print(colorize("\n╔══════════════════════════════════════════════════════════════╗", Colors.BLUE))
    print(colorize("║   ADVERSARIAL TRAJECTORY RESILIENCE - VISUALIZE MODE        ║", Colors.BLUE))
    print(colorize("╚══════════════════════════════════════════════════════════════╝\n", Colors.BLUE))
    
    width = args.width
    
    # If a log file is provided, load and visualize from it
    if args.log_file:
        try:
            with open(args.log_file, "r") as f:
                data = json.load(f)
            
            health_data = _extract_health_from_log(data)
            failures = _extract_failures_from_log(data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(colorize(f"Error loading log file: {e}", Colors.RED))
            return 1
    else:
        # Generate sample data for demonstration
        health_data, failures = _generate_sample_visualization()
    
    # Render the visualization
    _render_health_bar(health_data, failures, width)
    
    return 0


def _generate_sample_visualization() -> Tuple[List[Tuple[int, str]], List[int]]:
    """
    Generate sample data for visualization demonstration.
    
    Returns:
        Tuple of (health_data, failure_points)
    """
    # Simulate a trajectory with health transitions
    health_values = [
        TrajectoryHealth.HEALTHY,
        TrajectoryHealth.HEALTHY,
        TrajectoryHealth.HEALTHY,
        TrajectoryHealth.DEGRADED,    # Step 3 - failure injected
        TrajectoryHealth.DEGRADED,
        TrajectoryHealth.DEGRADED,
        TrajectoryHealth.CRITICAL,    # Step 6 - failure injected
        TrajectoryHealth.CRITICAL,
        TrajectoryHealth.RECOVERING,  # Recovery begins
        TrajectoryHealth.RECOVERING,
        TrajectoryHealth.HEALTHY,      # Fully recovered
        TrajectoryHealth.HEALTHY,
        TrajectoryHealth.DEGRADED,    # Step 12 - another issue
        TrajectoryHealth.DEGRADED,
        TrajectoryHealth.HEALTHY,      # Resolved
    ]
    
    health_data = [(i, h.value) for i, h in enumerate(health_values)]
    failure_points = [3, 6, 12]
    
    return health_data, failure_points


def _extract_health_from_log(data: Dict[str, Any]) -> List[Tuple[int, str]]:
    """Extract health data from a log file."""
    health_data = []
    
    if "branches" in data:
        for branch_id, branch_data in data["branches"].items():
            health = branch_data.get("health", "healthy")
            health_data.append((0, health))
    
    return health_data if health_data else [(0, "healthy")]


def _extract_failures_from_log(data: Dict[str, Any]) -> List[int]:
    """Extract failure injection points from a log file."""
    failures = []
    
    if "failures" in data:
        failures = data["failures"]
    
    return failures


def _render_health_bar(
    health_data: List[Tuple[int, str]],
    failure_points: List[int],
    width: int
) -> None:
    """Render the ASCII health bar visualization."""
    if not health_data:
        print(colorize("No health data available for visualization.", Colors.YELLOW))
        return
    
    print(f"{Colors.BOLD}Trajectory Health Visualization{Colors.RESET}")
    print(f"Bar width: {width} characters")
    print()
    
    # Map health values to characters and colors
    health_to_block = {
        "healthy": ("█", Colors.GREEN),
        "degraded": ("▓", Colors.YELLOW),
        "critical": ("▒", Colors.RED),
        "recovering": ("░", Colors.CYAN),
        "terminated": ("✗", Colors.RED),
    }
    
    # Normalize health data to fixed number of points
    max_steps = max(h[0] for h in health_data) + 1 if health_data else 1
    actual_steps = len(health_data)
    
    # Calculate how many health values per bar segment
    segment_size = max(1, actual_steps // width) if width < actual_steps else 1
    
    health_str = ""
    color_list = []
    
    for i in range(actual_steps):
        _, health_value = health_data[i]
        block, color = health_to_block.get(
            health_value,
            ("?", Colors.WHITE)
        )
        health_str += block
        color_list.append((len(health_str) - 1, color))
        
        # Check if this is a failure point
        step_num = health_data[i][0]
        if step_num in failure_points:
            health_str += colorize("✗", Colors.RED)
    
    # Print the health bar
    # Build colored segments
    print(f"{Colors.BOLD}HEALTH:{Colors.RESET} ", end="")
    
    segments = []
    for i, color in color_list:
        segments.append(color)
    
    # Simple approach: just print with color codes interspersed
    # For more precise coloring, we'd need a more complex approach
    print(_colorize_string(health_str, health_data, failure_points, health_to_block))
    
    # Print legend
    print()
    print(f"{Colors.BOLD}Legend:{Colors.RESET}")
    for health, (char, color) in health_to_block.items():
        if health != "terminated":
            print(f"  {colorize(char, color)} = {health.capitalize()}")
    
    if failure_points:
        print(f"  {colorize('✗', Colors.RED)} = Failure injection point")
    
    # Print status line
    print()
    final_health = health_data[-1][1] if health_data else "unknown"
    final_color = health_to_block.get(final_health, (None, Colors.WHITE))[1]
    
    try:
        health_enum = TrajectoryHealth(final_health)
        final_health_fmt = format_health(health_enum)
    except ValueError:
        final_health_fmt = colorize(f"[{final_health.upper()}]", Colors.WHITE)
    
    print(f"Final health: {final_health_fmt}")
    print(f"Total steps: {len(health_data)}")
    if failure_points:
        print(f"Failure points: {len(failure_points)}")


def _colorize_string(
    s: str,
    health_data: List[Tuple[int, str]],
    failure_points: List[int],
    health_to_block: Dict[str, Tuple[str, str]]
) -> str:
    """Create a colorized version of the health string."""
    # This is a simplified approach - for true ANSI coloring we'd need
    # to rebuild the string with embedded color codes
    
    result = ""
    step = 0
    
    for i, c in enumerate(s):
        if c == '✗':
            result += colorize(c, Colors.RED)
        elif c in "█▓▒░":
            # Determine which health this corresponds to
            data_idx = min(step, len(health_data) - 1)
            _, health_value = health_data[data_idx]
            _, color = health_to_block.get(health_value, ("?", Colors.WHITE))
            result += colorize(c, color)
            step += 1
        else:
            result += c
    
    return result


def main() -> int:
    """
    Main entry point for the CLI.
    
    Parses arguments and dispatches to the appropriate command handler.
    
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    parser = create_parser()
    args = parser.parse_args()
    
    # If no command provided, show help
    if args.command is None:
        parser.print_help()
        print()
        print(colorize("Use --help with a command for specific help.", Colors.DIM))
        return 0
    
    # Dispatch to command handler
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print(colorize("\n\nInterrupted by user.", Colors.YELLOW))
        return 130
    except Exception as e:
        print(colorize(f"\nError: {e}", Colors.RED))
        if "--verbose" in sys.argv or "-v" in sys.argv:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
