# Adversarial Trajectory Resilience

A framework for detecting, managing, and recovering from adversarial failures in LLM-based reasoning trajectories. It models reasoning chains as directed graphs, classifies failure modes in real-time, dynamically allocates compute budget across trajectory branches, and enforces verification checkpoints before committing to high-risk paths.

## Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌────────────────────────┐
│    Task     │───▶│ TrajectoryGraph  │───▶│ FailureModeDetector    │
│  (prompt)   │    │  (add_node/edge) │    │  (classify_state)      │
└─────────────┘    └──────────────────┘    └───────────┬────────────┘
                                                         │
                                                         ▼
                        ┌──────────────────┐    ┌────────────────────┐
                        │  VerificationGate│◀───│ UncertaintyEstimator│
                        │ (verify_transition│    │  (estimate_uncert.)│
                        │  create_checkpoint│    └────────────────────┘
                        └───────┬──────────┘              │
                                │                          │
                                ▼                          ▼
┌─────────────┐    ┌──────────────────┐    ┌────────────────────┐
│   Output    │◀───│ ResilienceOrchestrator│◀──│  ComputeAllocator │
│  (result)   │    │ (step / run_pipeline)│   │  (allocate)       │
└─────────────┘    └──────────────────┘    └────────────────────┘
```

### Pipeline Flow

```
[Task] → MONITOR → DETECT → PRUNE → ALLOCATE → VERIFY → [Next Task / Checkpoint]
              │        │        │        │         │
              ▼        ▼        ▼        ▼         ▼
        Update graph  Classify  Mark    Scale    Verify +
        + metrics   failure  prune    compute  checkpoint
                    modes   branches budget
```

**Health State Machine:**

```
HEALTHY ──▶ DEGRADED ──▶ CRITICAL ──▶ TERMINATED
   ▲              │                          │
   └──────────────┴────── RECOVERING ◀────────┘
```

## Core Components

### TrajectoryGraph (`src/trajectory.py`)

Models a reasoning trajectory as a directed graph. Each reasoning step is a node; transitions between steps are edges with optional weights.

- `add_node(node_id, metadata)` — Register a trajectory state
- `add_edge(start_node, end_node, weight)` — Record a transition
- `detect_cycles()` — Find loops in the reasoning chain (indicator of Reasoning Fatigue)
- `find_dead_ends()` — Identify leaf nodes with no outgoing edges

### FailureModeDetector (`src/detector.py`)

Classifies each reasoning state into one of five known adversarial failure modes using heuristic keyword and pattern detection.

- `classify_state(content, trajectory_history)` → `FailureClassification`

See **Failure Modes** below for the full list.

### VerificationGate (`src/verification.py`)

LawThinker-inspired safety layer that runs a binary checklist on every state transition before it is committed.

- `verify_transition(state, transition)` → `VerificationResult`
- `create_checkpoint(state)` → `Checkpoint` (immutable, checksum-verified)
- `rollback_to_checkpoint(checkpoint_id)` — Restore a verified state

Critical failures (confidence below threshold, suspicious content patterns, active failure modes) block the transition.

### ComputeAllocator (`src/allocator.py`)

Dynamically scales compute budget per trajectory branch based on real-time uncertainty and failure risk.

- `allocate(branch_id, state)` → `AllocationResult` — allocate more compute on high-uncertainty branches; reduce on high-risk ones
- `should_terminate(branch_id)` → `bool` — recommend termination when risk exceeds threshold or budget depletes
- `get_termination_reason(branch_id)` → `str`

### ResilienceOrchestrator (`src/orchestrator.py`)

The main pipeline coordinator. Wires together all components and drives the five-stage pipeline per reasoning step.

- `initialize(...)` — Wire together all components
- `step(state, branch_id, stage)` — Execute one pipeline stage
- `run_pipeline(state, branch_id)` — Execute the full pipeline
- `get_health(branch_id)` → `TrajectoryHealth` — query the state machine
- `needs_rollback(branch_id)` — `bool`
- `rollback_to_safe(branch_id)` — Restore from the last verified checkpoint

### MockAgent (`src/mock_agent.py`)

A deterministic mock reasoning agent for testing and demo purposes. Provides realistic reasoning responses and allows programmatic injection of failure modes without requiring real LLM calls.

- `step(prompt)` → `ReasoningStep` — one reasoning step
- `inject_failure(failure_mode, remaining_steps)` — activate a failure mode
- `get_trajectory()` → `List[ReasoningStep]`

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/adversarial-trajectory-resilience.git
cd adversarial-trajectory-resilience

# Create a virtual environment
python3 -m venv venv && source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Requirements** (`requirements.txt`):
```
numpy
pytest
torch
transformers
scikit-learn
```

## Usage

### Quick Start

```python
from src.mock_agent import MockAgent
from src.orchestrator import ResilienceOrchestrator, PipelineStage
from src.detector import FailureMode

# Create and initialize orchestrator with all components
orchestrator = ResilienceOrchestrator(auto_initialize=True)
agent = MockAgent()

# Inject a failure mode
agent.inject_failure(FailureMode.SELF_DOUBT, remaining_steps=2)

# Run a reasoning step
step = agent.step("Analyze this problem step by step.")
state = {
    "id": "state_0",
    "content": step.content,
    "confidence": 0.9,
    "history": [],
}

# Run the full resilience pipeline
results = orchestrator.run_pipeline(state, branch_id="main")

# Check trajectory health
health = orchestrator.get_health("main")
print(f"Health: {health.value}")

# See stage-by-stage results
for r in results:
    status = "✓" if r.success else "✗"
    print(f"  {status} {r.stage.value}: {r.data}")
```

### Using Each Component Individually

```python
# --- TrajectoryGraph ---
from src.trajectory import TrajectoryGraph

graph = TrajectoryGraph()
graph.add_node("step_0", {"content": "Let me analyze..."})
graph.add_node("step_1", {"content": "The key insight is..."})
graph.add_edge("step_0", "step_1")

cycles = graph.detect_cycles()
dead_ends = graph.find_dead_ends()

# --- FailureModeDetector ---
from src.detector import FailureModeDetector, FailureMode

detector = FailureModeDetector(threshold=0.5)
classification = detector.classify_state(
    "Actually, perhaps I was wrong earlier. Let me re-evaluate.",
    trajectory_history=["Let me analyze this problem."]
)
print(f"Mode: {classification.mode.value}, confidence: {classification.confidence:.2f}")

# --- VerificationGate ---
from src.verification import VerificationGate

gate = VerificationGate(confidence_threshold=0.7)
state = {"id": "step_0", "content": "The answer is 42.", "confidence": 0.8}

result = gate.verify_transition(state)
print(f"Verification passed: {result.passed}")
for check in result.checks:
    print(f"  [{check.result.value}] {check.name}: {check.details}")

checkpoint = gate.create_checkpoint(state)
print(f"Checkpoint ID: {checkpoint.checkpoint_id}")

# --- ComputeAllocator ---
from src.allocator import ComputeAllocator

allocator = ComputeAllocator(total_budget=500.0, base_allocation=50.0)
state = {"content": "I am not sure about this...", "confidence": 0.6}

result = allocator.allocate("branch_1", state=state)
print(f"Allocated: {result.amount} (success: {result.success})")
print(f"Should terminate: {allocator.should_terminate('branch_1')}")
```

### Running the Benchmark Suite

```python
from src.benchmark import BenchmarkRunner, BenchmarkScenario

runner = BenchmarkRunner(seed=42, verbose=True)
results = runner.run_full_benchmark()   # Runs clean + adversarial + high_uncertainty

runner.print_results(results)
```

## CLI

The CLI provides four commands: `monitor`, `analyze`, `benchmark`, and `visualize`.

```bash
# Real-time monitoring with failure injection
python -m src.cli monitor --steps 20 --inject-failures --interval 0.5

# Run benchmarks (clean, adversarial, high-uncertainty scenarios)
python -m src.cli benchmark --scenario adversarial --json --output results.json

# Analyze a saved trajectory log
python -m src.cli analyze trajectory_export.json --verbose

# ASCII visualization of health over time
python -m src.cli visualize --width 40
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `monitor` | Live simulation with MockAgent. Shows health transitions and failure detection per step. |
| `analyze <log_file>` | Load and inspect a trajectory log exported from `orchestrator.export_state()`. |
| `benchmark` | Run the full benchmark suite comparing baseline vs. resilience-enabled behavior. |
| `visualize` | ASCII bar chart of trajectory health over time. |

## Failure Modes

The framework detects and classifies five adversarial failure modes:

### 1. SELF_DOUBT
**Description:** The model second-guesses its own reasoning, repeatedly hedging or retracting conclusions.  
**Markers:** `"I am not sure"`, `"actually"`, `"wait"`, `"perhaps I was wrong"`, `"re-evaluating"`  
**Severity:** Moderate–High (degrades final answer quality without necessarily causing dangerous behavior)

### 2. SOCIAL_CONFORMITY
**Description:** The model abandons factual reasoning in favor of agreeing with the user's stated or implied position.  
**Markers:** `"as you suggested"`, `"as you mentioned"`, `"you're right"`, `"correctly pointed out"`  
**Severity:** High (can propagate user biases into critical outputs)

### 3. SUGGESTION_HIJACKING
**Description:** An external suggestion (from context, few-shot examples, or adversarial prompts) derails the model's reasoning chain.  
**Markers:** `"instead"` + `"actually"` in close proximity  
**Severity:** High (breaks logical continuity of the trajectory)

### 4. EMOTIONAL_SUSCEPTIBILITY
**Description:** The model responds to emotional urgency cues rather than reasoning through the problem.  
**Markers:** `"urgent"`, `"immediately"`, `"sorry"`, `"apologize"`, `"must act"`  
**Severity:** Moderate–High (can cause rushed, incorrect conclusions under time pressure)

### 5. REASONING_FATIGUE
**Description:** Repetitive or degenerate reasoning where the model produces increasingly simple or identical outputs.  
**Detection:** Three consecutive identical or near-identical reasoning states, or sharp drop in output complexity.  
**Severity:** High (leads to dead-end trajectories with no useful progress)

## Architecture Decisions

### Why a Graph-Based Trajectory Model?
Representing the reasoning trajectory as a directed graph (rather than a flat sequence) enables cycle detection and multi-branch trajectory tracking. This is essential because real adversarial prompts can cause the model to loop back on itself or explore multiple alternative paths.

### Why a Separate VerificationGate?
Separating verification from failure detection follows the principle of defense in depth. The detector identifies *what* is wrong; the gate enforces *whether* the state can proceed. This allows independent tuning of detection sensitivity vs. verification strictness.

### Why Dynamic Compute Allocation?
A fixed compute budget per trajectory is inefficient. High-uncertainty branches need more compute to resolve; high-risk branches need less (to limit exposure). The `ComputeAllocator` balances these competing needs using a priority score derived from uncertainty and risk.

### Why a Health State Machine?
The `TrajectoryHealth` state machine (`HEALTHY → DEGRADED → CRITICAL → TERMINATED`, with `RECOVERING` as a transient state) provides a clear, auditable policy for when to intervene. Recovery is always possible from `DEGRADED` or `CRITICAL`; `TERMINATED` is a hard stop. This makes the system's behavior predictable and debuggable.

### Why Immutable Checkpoints?
Checkpoints use SHA-256 checksums to guarantee integrity. Once created, a checkpoint cannot be silently modified. This is critical for auditability and for safe rollback — you never accidentally restore corrupted or adversarial state.
