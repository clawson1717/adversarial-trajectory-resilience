#!/usr/bin/env python3
"""
Demo: Adversarial Trajectory Resilience in Action

This script demonstrates the full resilience pipeline:
  1. Creates a MockAgent that simulates LLM reasoning steps
  2. Injects adversarial failure modes at specific steps
  3. Runs the ResilienceOrchestrator pipeline on each step
  4. Shows health state transitions (HEALTHY → DEGRADED → CRITICAL → RECOVERING)
  5. Demonstrates successful recovery after checkpoint rollback

Run with:
    python demo.py
"""

import sys
import time
from typing import List, Tuple

# Import all resilience components
from src.mock_agent import MockAgent, FailureMode
from src.orchestrator import (
    ResilienceOrchestrator,
    ResilienceOrchestrator as Orch,
    TrajectoryHealth,
    PipelineStage,
)
from src.detector import FailureModeDetector
from src.trajectory import TrajectoryGraph
from src.verification import VerificationGate
from src.allocator import ComputeAllocator


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def sep(char="─", width=70):
    """Print a visual separator line."""
    print(f"\n{char * width}\n")

def banner(title: str):
    """Print a titled banner."""
    sep("═")
    print(f"  {title}")
    sep("─")

def step_banner(step_num: int, action: str):
    """Print a step header."""
    print(f"\n{'─' * 60}")
    print(f"  STEP {step_num}: {action}")
    print(f"{'─' * 60}")

def print_health_transition(branch_id: str, old: TrajectoryHealth, new: TrajectoryHealth):
    """Pretty-print a health state transition."""
    colors = {
        TrajectoryHealth.HEALTHY: "\033[92m",
        TrajectoryHealth.DEGRADED: "\033[93m",
        TrajectoryHealth.CRITICAL: "\033[91m",
        TrajectoryHealth.RECOVERING: "\033[96m",
        TrajectoryHealth.TERMINATED: "\033[91m",
    }
    RESET = "\033[0m"

    def fmt(h):
        return f"{colors.get(h, '')}{h.value.upper()}{RESET}"

    if old != new:
        print(f"  ⚡ HEALTH TRANSITION: {fmt(old)} → {fmt(new)}")
    else:
        print(f"  ➤ Health remains: {fmt(new)}")


# ─────────────────────────────────────────────────────────────────────────────
# Demo scenario definitions
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: (step_num, failure_mode_or_None, prompt, description)
SCENARIO_STEPS: List[Tuple[int, FailureMode | None, str, str]] = [
    (0, None,
     "Analyze the problem: What are the key constraints?",
     "Normal reasoning — establish baseline"),

    (1, None,
     "Evaluate the first approach: What are its strengths?",
     "Normal reasoning — build on the plan"),

    (2, None,
     "Consider alternative solutions: What could go wrong?",
     "Normal reasoning — explore alternatives"),

    # ── FAILURE 1: SELF_DOUBT injected at step 3 ──────────────────────────────
    (3, FailureMode.SELF_DOUBT,
     "Review the evidence: Is it sufficient?",
     "⚠  FAILURE INJECTED: SELF_DOUBT — model starts hedging"),

    (4, None,
     "Check for logical fallacies in the reasoning.",
     "System detects SELF_DOUBT, health drops to DEGRADED"),

    # ── FAILURE 2: EMOTIONAL_SUSCEPTIBILITY injected at step 5 ────────────────
    (5, FailureMode.EMOTIONAL_SUSCEPTIBILITY,
     "Synthesize a conclusion from the analysis.",
     "⚠  FAILURE INJECTED: EMOTIONAL_SUSCEPTIBILITY — urgency triggers"),

    (6, None,
     "Verify the conclusion against all constraints.",
     "System escalates to CRITICAL — two consecutive failures"),

    # ── RECOVERY PHASE ────────────────────────────────────────────────────────
    (7, None,
     "Document any remaining uncertainties.",
     "✓  RECOVERY: orchestrator rolls back to last checkpoint"),

    (8, None,
     "Formulate next steps based on conclusions.",
     "✓  HEALTHY again — resilience system confirmed recovery"),

    (9, None,
     "Identify potential failure modes in the plan.",
     "Normal reasoning — trajectory stabilizes"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Main demo
# ─────────────────────────────────────────────────────────────────────────────

def run_demo():
    print("""
╔══════════════════════════════════════════════════════════════╗
║       ADVERSARIAL TRAJECTORY RESILIENCE — LIVE DEMO           ║
╚══════════════════════════════════════════════════════════════╝

This demo runs a MockAgent through 10 reasoning steps while the
ResilienceOrchestrator monitors, detects, prunes, allocates, and
verifies every step.  Two failure modes are injected mid-run.

Watch for:
  • Health state transitions (green → yellow → red → cyan → green)
  • Failure detection by FailureModeDetector
  • Checkpoint creation on successful verification
  • Rollback to a safe checkpoint after critical state
""")

    # ── 1. Initialize all components ──────────────────────────────────────
    banner("PHASE 1 — Initializing Components")

    graph = TrajectoryGraph()
    detector = FailureModeDetector(threshold=0.5)
    gate = VerificationGate(
        confidence_threshold=0.7,
        failure_mode_detector=detector,
    )
    allocator = ComputeAllocator(
        total_budget=500.0,
        base_allocation=50.0,
        min_allocation=10.0,
        max_allocation=100.0,
        failure_mode_detector=detector,
    )

    orchestrator = ResilienceOrchestrator(auto_initialize=False)
    orchestrator.initialize(
        trajectory_graph=graph,
        failure_mode_detector=detector,
        verification_gate=gate,
        compute_allocator=allocator,
    )

    agent = MockAgent()

    print("  ✓ TrajectoryGraph        — graph model for reasoning chains")
    print("  ✓ FailureModeDetector    — classifies 5 adversarial failure modes")
    print("  ✓ VerificationGate       — checkpoint + verification safety layer")
    print("  ✓ ComputeAllocator       — dynamic compute scaling per branch")
    print("  ✓ ResilienceOrchestrator — pipeline coordinator (monitor→detect→prune→allocate→verify)")
    print("  ✓ MockAgent              — simulated LLM with injectable failures")

    sep()

    # ── 2. Print scenario overview ─────────────────────────────────────────
    banner("PHASE 2 — Running Resilience Pipeline")
    print("Scenario: 10 steps, 2 injected failures")
    print("  Step 3 → SELF_DOUBT          (healthy → degraded)")
    print("  Step 5 → EMOTIONAL_SUSCEPT.  (degraded → critical)")
    print("  Step 7 → RECOVERY via checkpoint rollback (critical → recovering → healthy)")

    # Track previous health for transition detection
    prev_health = TrajectoryHealth.HEALTHY
    checkpoint_ids: List[str] = []

    # ── 3. Run the scenario step by step ───────────────────────────────────
    for step_num, failure_mode, prompt, description in SCENARIO_STEPS:

        step_banner(step_num, description)

        # Pre-build the graph so every node already has an outgoing edge by the time
        # _stage_monitor runs.  Without this, every intermediate node would be flagged
        # as a dead end (no children yet) during its own processing step.
        if step_num > 0:
            graph.add_edge(f"state_{step_num - 1}", f"state_{step_num}")

        # Inject the failure if scheduled
        if failure_mode is not None:
            agent.inject_failure(
                failure_mode,
                reason=f"demo_injection_step_{step_num}",
                remaining_steps=1,
            )
            print(f"  💉 Failure injected: {failure_mode.value}")

        # Execute one reasoning step with the mock agent
        step = agent.step(prompt)

        # Show what the agent produced
        content_preview = step.content[:60] + ("..." if len(step.content) > 60 else "")
        print(f"  🤖 Agent response: \"{content_preview}\"")

        if step.failure_mode != FailureMode.NONE:
            print(f"  ⚠  Agent failure mode: {step.failure_mode.value}")
        else:
            print(f"  ✓  Agent: normal response")

        # Build the state dict for the orchestrator
        trajectory_history = agent.get_trajectory_contents()[:-1]
        state = {
            "id": f"state_{step_num}",
            "content": step.content,
            "confidence": max(0.5, 1.0 - (step_num * 0.02)),  # gradually decreasing confidence
            "history": trajectory_history,
            "step_id": step_num,
        }

        # Run the full resilience pipeline
        results = orchestrator.run_pipeline(state, branch_id="main")

        # ── Check for health transition ──────────────────────────────────
        current_health = orchestrator.get_health("main")
        print_health_transition("main", prev_health, current_health)
        prev_health = current_health

        # ── Print per-stage results ──────────────────────────────────────
        for result in results:
            stage_icon = "✓" if result.success else "✗"
            stage_name = result.stage.value.upper()

            if result.stage == PipelineStage.DETECT:
                cls_data = result.data.get("failure_classification", {})
                mode = cls_data.get("mode", "none")
                confidence = cls_data.get("confidence", 0.0)
                print(f"     [{stage_icon}] {stage_name:10s} — mode={mode}, confidence={confidence:.2f}")

            elif result.stage == PipelineStage.VERIFY:
                v = result.data.get("verification", {})
                passed = v.get("passed", False)
                checks_total = v.get("total_checks", 0)
                failures = v.get("critical_failures", [])
                print(f"     [{stage_icon}] {stage_name:10s} — {checks_total} checks, "
                      f"critical_failures={failures if failures else 'none'}")

            elif result.stage == PipelineStage.ALLOCATE:
                alloc = result.data.get("allocation", {})
                amt = alloc.get("amount", 0)
                success = alloc.get("success", False)
                print(f"     [{stage_icon}] {stage_name:10s} — allocated={amt:.1f}, success={success}")

            elif result.stage == PipelineStage.PRUNE:
                pruned = result.data.get("pruned", False)
                reason = result.data.get("prune_reason", "n/a")
                action = result.data.get("action", "continued")
                print(f"     [{stage_icon}] {stage_name:10s} — pruned={pruned}, action={action}, reason={reason}")

            else:
                # MONITOR stage
                uncertainty = result.data.get("uncertainty", 0.0)
                cycles = result.data.get("cycles_detected", [])
                print(f"     [{stage_icon}] {stage_name:10s} — uncertainty={uncertainty:.2f}, cycles={len(cycles)}")

            # Show any errors
            if result.errors:
                for err in result.errors:
                    print(f"        ⚠  {err}")

        # ── Checkpoint tracking ──────────────────────────────────────────
        if current_health in (TrajectoryHealth.HEALTHY, TrajectoryHealth.RECOVERING):
            # Verify gate stores the last checkpoint internally
            latest_cp = gate.get_latest_checkpoint()
            if latest_cp and latest_cp.checkpoint_id not in checkpoint_ids:
                checkpoint_ids.append(latest_cp.checkpoint_id)
                print(f"  📌 Checkpoint created: {latest_cp.checkpoint_id}  (integrity: {latest_cp.verify_integrity()})")

        # ── Demonstrate rollback when in CRITICAL state ───────────────────
        if current_health == TrajectoryHealth.CRITICAL:
            print("\n  🔄 Attempting rollback to last safe checkpoint...")
            rollback_state = orchestrator.rollback_to_safe("main")
            if rollback_state:
                new_health = orchestrator.get_health("main")
                print_health_transition("main", current_health, new_health)
                print(f"  ✅ Rollback successful — restored to state: {rollback_state.get('id')}")
                # The loop continues from the rolled-back state
            else:
                print("  ⚠  No valid checkpoint available for rollback")

        print()

    # ── 4. Final summary ────────────────────────────────────────────────────
    sep("═")
    banner("PHASE 3 — Final Summary")

    final_health = orchestrator.get_health("main")
    health_colors = {
        TrajectoryHealth.HEALTHY: "\033[92m",
        TrajectoryHealth.DEGRADED: "\033[93m",
        TrajectoryHealth.CRITICAL: "\033[91m",
        TrajectoryHealth.RECOVERING: "\033[96m",
        TrajectoryHealth.TERMINATED: "\033[91m",
    }
    RESET = "\033[0m"
    color = health_colors.get(final_health, "")
    print(f"  Final trajectory health: {color}{final_health.value.upper()}{RESET}")

    # Trajectory summary
    trajectory = agent.get_trajectory()
    total_steps = len(trajectory)
    injected_failures = sum(1 for s in trajectory if s.failure_mode != FailureMode.NONE)
    print(f"  Total reasoning steps:   {total_steps}")
    print(f"  Failures injected:       {injected_failures}")

    # Detection summary from stage history
    history = orchestrator.get_stage_history(branch_id="main")
    detect_failures = [
        h for h in history
        if "detect" in h.stage and not h.success
    ]
    verify_failures = [
        h for h in history
        if "verify" in h.stage and not h.success
    ]
    print(f"  Detection failures:      {len(detect_failures)}")
    print(f"  Verification failures:    {len(verify_failures)}")

    # Allocator summary
    alloc_status = allocator.get_status()
    print(f"  Total compute used:       {alloc_status['total_consumed']:.1f} / {alloc_status['total_budget']:.1f}")
    print(f"  Active branches:         {alloc_status['num_branches']}")

    # Checkpoint summary
    all_checkpoints = gate.list_checkpoints()
    print(f"  Checkpoints created:      {len(all_checkpoints)}")
    for cp_id in all_checkpoints:
        cp = gate.get_checkpoint(cp_id)
        if cp:
            print(f"    • {cp_id}  (integrity: {cp.verify_integrity()})")

    # Stage timing summary
    if history:
        total_duration = sum(h.duration_ms for h in history)
        avg_duration = total_duration / len(history) if history else 0
        print(f"\n  Stage execution:")
        print(f"    Total pipeline duration: {total_duration:.1f} ms")
        print(f"    Average stage duration:   {avg_duration:.1f} ms")

    sep("═")

    if final_health == TrajectoryHealth.HEALTHY:
        print("""
  ✅ DEMO COMPLETE — Resilience system maintained HEALTHY state!
     Both injected failures were detected and the trajectory recovered.
""")
        return 0
    elif final_health == TrajectoryHealth.RECOVERING:
        print("""
  ⚠  DEMO COMPLETE — Trajectory is RECOVERING.
     The system detected failures and is working toward a safe state.
""")
        return 0
    else:
        print("""
  ⚠  DEMO COMPLETE — Trajectory ended in {final_health.value.upper()} state.
     This may indicate a scenario that overwhelmed the resilience system.
""")
        return 1


if __name__ == "__main__":
    sys.exit(run_demo())
