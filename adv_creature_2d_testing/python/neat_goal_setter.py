import random
import math
import torch
from typing import List, Dict, Tuple, Any

# Activation functions for NEAT nodes
def sigmoid_act(val, bias):
    x = val + bias
    return 1.0 / (1.0 + math.exp(-max(min(x, 20.0), -20.0)))

def sine_act(val, bias):
    return math.sin(val + bias)

def relu_act(val, bias):
    return max(val + bias, 0.0)

def tanh_act(val, bias):
    return math.tanh(val + bias)

def linear_act(val, bias):
    return val + bias

def gaussian_act(val, bias):
    x = val + bias
    return math.exp(-(x * x))

NEURON_ACTIVATIONS = {
    "sigmoid": sigmoid_act,
    "sine": sine_act,
    "relu": relu_act,
    "tanh": tanh_act,
    "linear": linear_act,
    "gaussian": gaussian_act
}

class NEATGoalSetter:
    """
    NEAT-Evolved Goal-Setter Model (M1) (§2 & §8).
    
    Inputs: Concatenated 5-frame memory buffer of global variables
            (position, velocity, energy, dopamine, vision, relative food/enemy coordinates).
    Outputs: 3D Goal Vector g_t = [g_x, g_y, g_z].
    
    Evolves via structural mutations (add node, add connection, weight/bias perturbation).
    """
    def __init__(self, input_dim: int = 50, output_dim: int = 3):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nodes: List[Dict[str, Any]] = []
        self.connections: List[Dict[str, Any]] = []
        self.innovation_counter = 0

        self._build_initial_genome()

    def _build_initial_genome(self):
        self.nodes = []
        self.connections = []
        
        # 1. Create Input Nodes (IDs 1 .. input_dim)
        for i in range(1, self.input_dim + 1):
            self.nodes.append({
                "id": i,
                "type": "input",
                "activation": "linear",
                "bias": 0.0
            })
            
        # 2. Create Output Nodes (IDs input_dim + 1 .. input_dim + output_dim)
        for j in range(1, self.output_dim + 1):
            out_id = self.input_dim + j
            self.nodes.append({
                "id": out_id,
                "type": "output",
                "activation": "tanh",
                "bias": 0.0
            })
            
        # 3. Create initial fully connected edges between inputs and outputs
        for i in range(1, self.input_dim + 1):
            for j in range(1, self.output_dim + 1):
                out_id = self.input_dim + j
                self.innovation_counter += 1
                self.connections.append({
                    "id": self.innovation_counter,
                    "in": i,
                    "out": out_id,
                    "weight": random.gauss(0, 0.5),
                    "enabled": True
                })

    def forward(self, input_vector: List[float]) -> torch.Tensor:
        """
        Runs topological forward pass over NEAT network graph using 5-frame input vector.
        """
        # Ensure input dimensions match or pad with zeros if needed
        if len(input_vector) < self.input_dim:
            padded_input = input_vector + [0.0] * (self.input_dim - len(input_vector))
        else:
            padded_input = input_vector[:self.input_dim]

        node_outputs = {}
        # Set input node outputs
        for i in range(1, self.input_dim + 1):
            node_outputs[i] = padded_input[i - 1]

        # Evaluate hidden and output nodes
        # Sort nodes by ID order
        sorted_nodes = sorted([n for n in self.nodes if n["type"] != "input"], key=lambda x: x["id"])
        
        for node in sorted_nodes:
            nid = node["id"]
            act_fn = NEURON_ACTIVATIONS.get(node["activation"], tanh_act)
            bias = node["bias"]
            
            # Sum incoming connected edges
            sum_val = 0.0
            for conn in self.connections:
                if conn["enabled"] and conn["out"] == nid:
                    in_val = node_outputs.get(conn["in"], 0.0)
                    sum_val += in_val * conn["weight"]
                    
            node_outputs[nid] = act_fn(sum_val, bias)

        # Extract 3D goal output (output node IDs: input_dim + 1, input_dim + 2, input_dim + 3)
        goal_x = node_outputs.get(self.input_dim + 1, 0.0)
        goal_y = node_outputs.get(self.input_dim + 2, 0.0)
        goal_z = node_outputs.get(self.input_dim + 3, 0.0)
        
        return torch.tensor([goal_x, goal_y, goal_z], dtype=torch.float32)

    def mutate(self, weight_prob: float = 0.8, add_node_prob: float = 0.05, add_conn_prob: float = 0.1):
        """
        Mutates NEAT genome parameters (weights, biases, topology).
        """
        # 1. Mutate existing connection weights & node biases
        if random.random() < weight_prob:
            for conn in self.connections:
                if random.random() < 0.2:
                    conn["weight"] += random.gauss(0, 0.2)
            for node in self.nodes:
                if random.random() < 0.2:
                    node["bias"] += random.gauss(0, 0.1)

        # 2. Mutate: Add Node (Splits an existing connection)
        if random.random() < add_node_prob and len(self.connections) > 0:
            enabled_conns = [c for c in self.connections if c["enabled"]]
            if enabled_conns:
                target_conn = random.choice(enabled_conns)
                target_conn["enabled"] = False
                
                new_node_id = max([n["id"] for n in self.nodes]) + 1
                self.nodes.append({
                    "id": new_node_id,
                    "type": "hidden",
                    "activation": random.choice(["tanh", "sine", "relu", "sigmoid"]),
                    "bias": 0.0
                })
                
                # Connection in -> new_node (weight = 1.0)
                self.innovation_counter += 1
                self.connections.append({
                    "id": self.innovation_counter,
                    "in": target_conn["in"],
                    "out": new_node_id,
                    "weight": 1.0,
                    "enabled": True
                })
                
                # Connection new_node -> out (weight = old weight)
                self.innovation_counter += 1
                self.connections.append({
                    "id": self.innovation_counter,
                    "in": new_node_id,
                    "out": target_conn["out"],
                    "weight": target_conn["weight"],
                    "enabled": True
                })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "innovation_counter": self.innovation_counter,
            "nodes": self.nodes,
            "connections": self.connections
        }

    def load_dict(self, state_dict: Dict[str, Any]):
        self.input_dim = state_dict.get("input_dim", self.input_dim)
        self.output_dim = state_dict.get("output_dim", self.output_dim)
        self.innovation_counter = state_dict.get("innovation_counter", self.innovation_counter)
        self.nodes = state_dict.get("nodes", [])
        self.connections = state_dict.get("connections", [])
