"""
NEAT Goal-Setter (M1) — §2, §8 of Creature Brain Architecture.

Inputs:  Fixed-size live global-state vector (energy, motion, nearby entities,
         and previous displacement goal). The live ecosystem uses 28 values.
Outputs: 2-D displacement goal  g_t = [g_x, g_y]  (tanh-clamped).

Evolves via structural mutations:
  • Weight/bias perturbation
  • Add node  (weight-preserving split — identical to Python prototype §5.2)
  • Add connection  (feed-forward only, cycle-checked)
"""

import random
import math
import torch
from collections import deque
from typing import List, Dict, Tuple, Any, Optional

# ---------------------------------------------------------------------------
# Activation functions (scalar; same set as Python prototype's brain.py)
# ---------------------------------------------------------------------------

def _clamp(x: float, lo: float = -20.0, hi: float = 20.0) -> float:
    return max(lo, min(hi, x))

def sigmoid_act(val: float, bias: float) -> float:
    return 1.0 / (1.0 + math.exp(-_clamp(val + bias)))

def sine_act(val: float, bias: float) -> float:
    return math.sin(val + bias)

def relu_act(val: float, bias: float) -> float:
    return max(val + bias, 0.0)

def tanh_act(val: float, bias: float) -> float:
    return math.tanh(val + bias)

def linear_act(val: float, bias: float) -> float:
    return val + bias

def gaussian_act(val: float, bias: float) -> float:
    x = val + bias
    return math.exp(-(x * x))

NEURON_ACTIVATIONS: Dict[str, Any] = {
    "sigmoid":  sigmoid_act,
    "sine":     sine_act,
    "relu":     relu_act,
    "tanh":     tanh_act,
    "linear":   linear_act,
    "gaussian": gaussian_act,
}

HIDDEN_ACTIVATIONS = ["tanh", "sine", "relu", "sigmoid"]


class NEATInnovationRegistry:
    """Historical markings shared by every M1 genome in one ecosystem."""
    def __init__(self, next_innovation: int = 1, next_node: int = 10000, connections=None, split_nodes=None):
        self.next_innovation, self.next_node = next_innovation, next_node
        self.connections = {} if connections is None else {tuple(map(int, k.split(","))): int(v) for k, v in connections.items()}
        self.split_nodes = {} if split_nodes is None else {int(k): int(v) for k, v in split_nodes.items()}
    def connection(self, src: int, dst: int) -> int:
        key = (src, dst)
        if key not in self.connections:
            self.connections[key] = self.next_innovation; self.next_innovation += 1
        return self.connections[key]
    def split_node(self, connection_id: int) -> int:
        if connection_id not in self.split_nodes:
            self.split_nodes[connection_id] = self.next_node; self.next_node += 1
        return self.split_nodes[connection_id]
    def to_dict(self):
        return {"next_innovation": self.next_innovation, "next_node": self.next_node,
                "connections": {f"{a},{b}": v for (a,b),v in self.connections.items()}, "split_nodes": self.split_nodes}
    @classmethod
    def from_dict(cls, value): return cls(value.get("next_innovation",1), value.get("next_node",10000), value.get("connections",{}), value.get("split_nodes",{}))


# ---------------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------------

def _build_adjacency(nodes: List[Dict], connections: List[Dict]) -> Dict[int, List[int]]:
    """Returns adjacency list (in -> [out, ...]) for enabled connections only."""
    adj: Dict[int, List[int]] = {n["id"]: [] for n in nodes}
    for c in connections:
        if c["enabled"]:
            adj[c["in"]].append(c["out"])
    return adj


def _topological_sort(nodes: List[Dict], connections: List[Dict]) -> List[int]:
    """
    Kahn's algorithm on the enabled subgraph.
    Returns a topological ordering of all node IDs.
    Raises RuntimeError if a cycle is detected (should not happen after
    _would_create_cycle checks, but defensive).
    """
    in_degree: Dict[int, int] = {n["id"]: 0 for n in nodes}
    adj = _build_adjacency(nodes, connections)
    for src, dsts in adj.items():
        for dst in dsts:
            in_degree[dst] = in_degree.get(dst, 0) + 1

    queue: deque[int] = deque()
    for nid, deg in in_degree.items():
        if deg == 0:
            queue.append(nid)

    order: List[int] = []
    while queue:
        nid = queue.popleft()
        order.append(nid)
        for dst in adj.get(nid, []):
            in_degree[dst] -= 1
            if in_degree[dst] == 0:
                queue.append(dst)

    if len(order) != len(nodes):
        raise RuntimeError(
            "NEATGoalSetter: cycle detected in genome graph — "
            f"only sorted {len(order)} of {len(nodes)} nodes."
        )
    return order


def _would_create_cycle(nodes: List[Dict], connections: List[Dict],
                         new_in: int, new_out: int) -> bool:
    """
    Returns True if adding edge new_in -> new_out would create a directed cycle.
    Uses DFS reachability: if new_out can already reach new_in, adding the edge
    would form a cycle.
    """
    adj = _build_adjacency(nodes, connections)
    # DFS from new_out; if we can reach new_in, it's a cycle.
    visited = set()
    stack = [new_out]
    while stack:
        cur = stack.pop()
        if cur == new_in:
            return True
        if cur in visited:
            continue
        visited.add(cur)
        stack.extend(adj.get(cur, []))
    return False


# ---------------------------------------------------------------------------
# NEATGoalSetter
# ---------------------------------------------------------------------------

class NEATGoalSetter:
    """
    NEAT-evolved goal-setter (M1).

    The genome is a list of node dicts and connection dicts:
      node:  {id, type∈{input,hidden,output}, activation, bias}
      conn:  {id, in, out, weight, enabled}

    Forward pass uses Kahn topological sort (not sort-by-ID) so it is
    correct even when hidden nodes are added in non-topological order.
    """

    def __init__(self, input_dim: int = 30, output_dim: int = 3, registry: Optional[NEATInnovationRegistry] = None):
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.nodes:       List[Dict[str, Any]] = []
        self.connections: List[Dict[str, Any]] = []
        self.registry = registry or NEATInnovationRegistry()
        self.innovation_counter = 0
        self._build_initial_genome()

    # ------------------------------------------------------------------
    # Genome construction
    # ------------------------------------------------------------------

    def _build_initial_genome(self):
        self.nodes       = []
        self.connections = []

        # Input nodes: IDs 1 .. input_dim
        for i in range(1, self.input_dim + 1):
            self.nodes.append({
                "id":         i,
                "type":       "input",
                "activation": "linear",
                "bias":       0.0,
            })

        # Output nodes: IDs input_dim+1 .. input_dim+output_dim
        for j in range(1, self.output_dim + 1):
            self.nodes.append({
                "id":         self.input_dim + j,
                "type":       "output",
                "activation": "tanh",
                "bias":       0.0,
            })

        # Initial fully-connected input → output edges
        for i in range(1, self.input_dim + 1):
            for j in range(1, self.output_dim + 1):
                out_id = self.input_dim + j
                self.innovation_counter = self.registry.connection(i, out_id)
                self.connections.append({
                    "id":      self.innovation_counter,
                    "in":      i,
                    "out":     out_id,
                    "weight":  random.gauss(0.0, 0.5),
                    "enabled": True,
                })

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, input_vector: List[float]) -> torch.Tensor:
        """
        Topological forward pass over the NEAT genome.
    Returns a tanh-bounded 2-D goal tensor [g_x, g_y] for the live game.
        """
        # Pad or truncate to match input_dim
        if len(input_vector) < self.input_dim:
            padded = list(input_vector) + [0.0] * (self.input_dim - len(input_vector))
        else:
            padded = list(input_vector[: self.input_dim])

        # Seed input node values
        node_values: Dict[int, float] = {}
        for i in range(1, self.input_dim + 1):
            node_values[i] = padded[i - 1]

        # Compute topological order (Kahn's algorithm)
        order = _topological_sort(self.nodes, self.connections)

        # Build enabled-connections lookup keyed by destination
        incoming: Dict[int, List[Tuple[int, float]]] = {}
        for c in self.connections:
            if c["enabled"]:
                incoming.setdefault(c["out"], []).append((c["in"], c["weight"]))

        # Evaluate non-input nodes in topological order
        node_by_id = {n["id"]: n for n in self.nodes}
        for nid in order:
            node = node_by_id[nid]
            if node["type"] == "input":
                continue  # already seeded
            act_fn = NEURON_ACTIVATIONS.get(node["activation"], tanh_act)
            raw = sum(
                node_values.get(src_id, 0.0) * w
                for src_id, w in incoming.get(nid, [])
            )
            node_values[nid] = act_fn(raw, node["bias"])

        # Output dimension is deliberately configurable: the Unity prototype is 2-D.
        return torch.tensor(
            [node_values.get(self.input_dim + index, 0.0) for index in range(1, self.output_dim + 1)],
            dtype=torch.float32,
        )

    # ------------------------------------------------------------------
    # Mutations (NEAT)
    # ------------------------------------------------------------------

    def mutate(
        self,
        weight_prob:    float = 0.8,
        add_node_prob:  float = 0.05,
        add_conn_prob:  float = 0.10,
        activation_prob: float = 0.03,
    ):
        """
        Three mutation types:
          1. Perturb weights/biases (Gaussian noise).
          2. Add node — weight-preserving split of an existing connection
             (new node gets identity activation so the network's output is
             unchanged immediately after the mutation, per §5.2).
          3. Add connection — random new enabled edge between two nodes,
             feed-forward only (cycle-checked).
        """
        # 1. Weight / bias perturbation
        if random.random() < weight_prob:
            for conn in self.connections:
                if random.random() < 0.2:
                    conn["weight"] += random.gauss(0.0, 0.2)
            for node in self.nodes:
                if random.random() < 0.2:
                    node["bias"] += random.gauss(0.0, 0.1)

        # 2. Add node  (weight-preserving split)
        if random.random() < add_node_prob and self.connections:
            enabled_conns = [c for c in self.connections if c["enabled"]]
            if enabled_conns:
                target_conn = random.choice(enabled_conns)
                target_conn["enabled"] = False

                new_id = self.registry.split_node(target_conn["id"])
                self.nodes.append({
                    "id":         new_id,
                    "type":       "hidden",
                    # Identity preserves the split connection's behaviour. A later
                    # activation mutation may turn it into a nonlinear hidden node.
                    "activation": "linear",
                    "bias":       0.0,
                })

                # in -> new (weight=1.0 preserves target_conn's magnitude)
                self.innovation_counter = self.registry.connection(target_conn["in"], new_id)
                self.connections.append({
                    "id":      self.innovation_counter,
                    "in":      target_conn["in"],
                    "out":     new_id,
                    "weight":  1.0,
                    "enabled": True,
                })

                # new -> out (weight = old weight, so composite = old weight)
                self.innovation_counter = self.registry.connection(new_id, target_conn["out"])
                self.connections.append({
                    "id":      self.innovation_counter,
                    "in":      new_id,
                    "out":     target_conn["out"],
                    "weight":  target_conn["weight"],
                    "enabled": True,
                })

        # 3. Add connection  (feed-forward only, checked for cycles)
        if random.random() < add_conn_prob:
            self._try_add_connection()

        # Activation changes are deliberately separate from add-node. New
        # nodes start linear so splitting an edge is initially identity-like.
        hidden = [node for node in self.nodes if node["type"] == "hidden"]
        if hidden and random.random() < activation_prob:
            node = random.choice(hidden)
            choices = [name for name in HIDDEN_ACTIVATIONS if name != node["activation"]]
            if choices:
                node["activation"] = random.choice(choices)

    def _try_add_connection(self, max_attempts: int = 20):
        """
        Attempt to add a new enabled connection between two nodes such that
        (a) the edge does not already exist as an enabled connection and
        (b) adding it would not create a directed cycle.
        Gives up after max_attempts trials to avoid infinite loops.
        """
        non_output = [n["id"] for n in self.nodes if n["type"] != "output"]
        non_input  = [n["id"] for n in self.nodes if n["type"] != "input"]

        if not non_output or not non_input:
            return  # degenerate genome; nothing to do

        existing_edges = {
            (c["in"], c["out"]) for c in self.connections if c["enabled"]
        }

        for _ in range(max_attempts):
            src = random.choice(non_output)
            dst = random.choice(non_input)
            if src == dst:
                continue
            if (src, dst) in existing_edges:
                continue
            if _would_create_cycle(self.nodes, self.connections, src, dst):
                continue

            self.innovation_counter = self.registry.connection(src, dst)
            self.connections.append({
                "id":      self.innovation_counter,
                "in":      src,
                "out":     dst,
                "weight":  random.gauss(0.0, 0.5),
                "enabled": True,
            })
            return  # success

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_dim":           self.input_dim,
            "output_dim":          self.output_dim,
            "innovation_counter":  self.innovation_counter,
            "nodes":               self.nodes,
            "connections":         self.connections,
        }

    def load_dict(self, state: Dict[str, Any]):
        self.input_dim          = state.get("input_dim",          self.input_dim)
        self.output_dim         = state.get("output_dim",         self.output_dim)
        self.innovation_counter = state.get("innovation_counter", self.innovation_counter)
        self.nodes              = state.get("nodes",              [])
        self.connections        = state.get("connections",        [])

    @classmethod
    def crossover(cls, first: "NEATGoalSetter", first_fitness: float,
                  second: "NEATGoalSetter", second_fitness: float,
                  registry: NEATInnovationRegistry) -> "NEATGoalSetter":
        """Innovation-aligned NEAT crossover; lower-level registry stays shared."""
        fitter = first if first_fitness >= second_fitness else second
        equal = first_fitness == second_fitness
        nodes_a, nodes_b = {n["id"]: n for n in first.nodes}, {n["id"]: n for n in second.nodes}
        conns_a, conns_b = {c["id"]: c for c in first.connections}, {c["id"]: c for c in second.connections}
        child = cls(first.input_dim, first.output_dim, registry)
        child.nodes = []
        for node_id in sorted(set(nodes_a) | set(nodes_b)):
            if node_id in nodes_a and node_id in nodes_b: source = random.choice((nodes_a[node_id], nodes_b[node_id]))
            elif equal: source = nodes_a.get(node_id, nodes_b.get(node_id))
            else: source = (nodes_a if fitter is first else nodes_b).get(node_id)
            if source is not None: child.nodes.append(dict(source))
        child.connections = []
        node_ids = {n["id"] for n in child.nodes}
        for innovation in sorted(set(conns_a) | set(conns_b)):
            if innovation in conns_a and innovation in conns_b:
                source = dict(random.choice((conns_a[innovation], conns_b[innovation])))
                if (not conns_a[innovation]["enabled"] or not conns_b[innovation]["enabled"]) and random.random() < .75: source["enabled"] = False
            elif equal: source = dict(conns_a.get(innovation, conns_b.get(innovation)))
            else: source = dict((conns_a if fitter is first else conns_b).get(innovation, {}))
            if source and source["in"] in node_ids and source["out"] in node_ids: child.connections.append(source)
        child.innovation_counter = max([c["id"] for c in child.connections] or [0])
        return child


def compatibility_distance(first: NEATGoalSetter, second: NEATGoalSetter) -> float:
    """NEAT-style structural/weight distance used for M1 subspecies."""
    a = {c["id"]: c for c in first.connections}
    b = {c["id"]: c for c in second.connections}
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    disjoint = sum(key not in a or key not in b for key in keys) / max(1, len(keys))
    matching = set(a) & set(b)
    weight = 0.0 if not matching else sum(abs(a[key]["weight"] - b[key]["weight"]) for key in matching) / len(matching)
    return disjoint + .4 * weight
