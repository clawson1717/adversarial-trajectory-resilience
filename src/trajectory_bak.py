from typing import Dict, List, Set, Optional, Any

class TrajectoryGraph:
    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.adj: Dict[str, List[str]] = {}
        self.weights: Dict[tuple, float] = {}

    def add_node(self, node_id: str, metadata: Optional[Dict[str, Any]] = None):
        if node_id not in self.nodes:
            self.nodes[node_id] = metadata or {}
            self.adj[node_id] = []

    def add_edge(self, start_node: str, end_node: str, weight: float = 1.0):
        if start_node not in self.nodes:
            self.add_node(start_node)
        if end_node not in self.nodes:
            self.add_node(end_node)
        
        if end_node not in self.adj[start_node]:
            self.adj[start_node].append(end_node)
            self.weights[(start_node, end_node)] = weight

    def detect_cycles(self) -> List[List[str]]:
        cycles = []
        visited = set()
        stack = []
        path = []

        def visit(node):
            if node in path:
                # Cycle detected
                cycle_start_idx = path.index(node)
                cycles.append(path[cycle_start_idx:] + [node])
                return
            
            if node in visited:
                return

            visited.add(node)
            path.append(node)
            for neighbor in self.adj.get(node, []):
                visit(neighbor)
            path.pop()

        for node in self.nodes:
            visit(node)
        
        return cycles

    def find_dead_ends(self) -> List[str]:
        return [node for node in self.nodes if not self.adj.get(node)]
