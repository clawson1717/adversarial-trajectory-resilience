import pytest
import sys
import os

# Ensure src is in paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from trajectory import TrajectoryGraph

def test_add_node():
    graph = TrajectoryGraph()
    graph.add_node("A", {"label": "start"})
    assert "A" in graph.nodes
    assert graph.nodes["A"]["label"] == "start"

def test_add_edge():
    graph = TrajectoryGraph()
    graph.add_edge("A", "B", 1.5)
    assert "A" in graph.nodes
    assert "B" in graph.nodes
    assert "B" in graph.adj["A"]
    assert graph.weights[("A", "B")] == 1.5

def test_detect_cycles():
    graph = TrajectoryGraph()
    graph.add_edge("A", "B")
    graph.add_edge("B", "C")
    graph.add_edge("C", "A")
    
    cycles = graph.detect_cycles()
    assert len(cycles) > 0
    # The current cycle detection might return multiple paths or subsets depending on structure,
    # but for a simple cycle it should find it.
    assert "A" in cycles[0]
    assert "B" in cycles[0]
    assert "C" in cycles[0]

def test_find_dead_ends():
    graph = TrajectoryGraph()
    graph.add_edge("A", "B")
    graph.add_node("C")
    
    dead_ends = graph.find_dead_ends()
    assert "B" in dead_ends
    assert "C" in dead_ends
    assert "A" not in dead_ends
