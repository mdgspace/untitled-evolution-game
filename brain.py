import random
import math
from random import randrange
import uuid

#for NEAT brains
innovation_number = 0


#================= BRAIN MATH NODES =================
#the entire brain deals with 3d vectors only. treating vectors as 3-tuples
#in this python mock-up; later, when translating to game engine code,
#swap these for the engine's native vector3 type.

def sigmoid(vector, bias):
    return (1/(1 + math.exp(-vector[0]-bias[0])), 1/(1 + math.exp(-vector[1]-bias[1])), 1/(1 + math.exp(-vector[2]-bias[2])))

def sine(vector, bias):
    return (math.sin(vector[0]+bias[0]), math.sin(vector[1]+bias[1]), math.sin(vector[2]+bias[2]))

def relu(vector, bias):
    return (max(vector[0]+bias[0], 0), max(vector[1]+bias[1], 0), max(vector[2]+bias[2], 0))

def gaussian(vector, bias):
    return (
        math.exp(-((vector[0]+bias[0])**2)),
        math.exp(-((vector[1]+bias[1])**2)),
        math.exp(-((vector[2]+bias[2])**2)),
    )

def linear(vector, bias):
    return (vector[0]+bias[0], vector[1]+bias[1], vector[2]+bias[2])

def tanh(vector, bias):
    return (math.tanh(vector[0]+bias[0]), math.tanh(vector[1]+bias[1]), math.tanh(vector[2]+bias[2]))

neuron_types = [sigmoid, sine, relu, gaussian, linear, tanh]


def topological_sort(nodes, connections):
    #kahn's algorithm over a brain-shaped graph: nodes is a list of dicts
    #with a "number" field, connections is a list of dicts with
    #"start"/"end"/"enabled". returns node numbers ordered so every
    #connection points from an earlier entry to a later one.
    #not called anywhere yet -- kept ready for when evaluate_nn needs it.
    in_degree = {n["number"]: 0 for n in nodes}
    adjacency = {n["number"]: [] for n in nodes}
    for conn in connections:
        if not conn["enabled"]:
            continue
        adjacency[conn["start"]].append(conn["end"])
        in_degree[conn["end"]] += 1
    queue = [num for num, deg in in_degree.items() if deg == 0]
    ordered = []
    while queue:
        current = queue.pop(0)
        ordered.append(current)
        for neighbor in adjacency[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    return ordered


#================= BRAIN =================

class brain():
    #NEAT-style brain: standard NEAT topology evolution (innovation
    #numbers, structural mutation), but every signal is a vector3 and each
    #node's nonlinearity is picked per-node from neuron_types.
    #
    #NOTE: this class deliberately never imports torso.py. __init__ takes
    #a torso-like object and only ever calls .get_all_limbs()/.name/.data
    #on it -- duck typing, not a hard dependency. this is what keeps
    #brain.py free of the torso<->brain circular import that would
    #otherwise exist (torso.py needs brain, but brain.py never needs torso).
    def __init__(self, torso, input_names=None):
        self.name = str(uuid.uuid4())
        self.torso = torso
        self.torso_id = torso.name

        if input_names is None:
            input_names = ["torso"]
        self.input_names = input_names

        self.all_limbs = torso.get_all_limbs()

        self.nodes = []
        node_number = 1

        for name in self.input_names:
            self.nodes.append({
                "number": node_number, "activation": random.choice(neuron_types),
                "bias": (0, 0, 0), "edge": name, "role": "input", "kind": "global",
                "depth": 0, "limb_gene_path": None, "node_innovation": None,
                "last_output": (0, 0, 0)
            })
            node_number += 1

        self.limb_output_numbers = {}
        global innovation_number
        self.brain_connections = []

        for limb_obj in self.all_limbs:
            output_number = node_number
            self.nodes.append({
                "number": output_number, "activation": random.choice(neuron_types),
                "bias": (0, 0, 0), "edge": limb_obj.name, "role": "output", "kind": "torque",
                "depth": limb_obj.depth, "limb_gene_path": limb_obj.gene_path,
                "node_innovation": None, "last_output": (0, 0, 0)
            })
            node_number += 1
            self.limb_output_numbers[limb_obj.name] = output_number

            joint_angle_number = node_number
            self.nodes.append({
                "number": joint_angle_number, "activation": random.choice(neuron_types),
                "bias": (0, 0, 0), "edge": limb_obj.name, "role": "input", "kind": "joint_angle",
                "depth": limb_obj.depth, "limb_gene_path": limb_obj.gene_path,
                "node_innovation": None, "last_output": (0, 0, 0)
            })
            node_number += 1

            touch_number = node_number
            self.nodes.append({
                "number": touch_number, "activation": random.choice(neuron_types),
                "bias": (0, 0, 0), "edge": limb_obj.name, "role": "input", "kind": "touch",
                "depth": limb_obj.depth, "limb_gene_path": limb_obj.gene_path,
                "node_innovation": None, "last_output": (0, 0, 0)
            })
            node_number += 1

            for local_input_number in (joint_angle_number, touch_number):
                innovation_number += 1
                self.brain_connections.append({
                    "innovation_number": innovation_number, "start": local_input_number,
                    "end": output_number, "enabled": True, "weight": (random.uniform(-2.0, 2.0),) * 3
                })

        global_input_nodes = [n for n in self.nodes if n["kind"] == "global"]
        output_nodes = [n for n in self.nodes if n["role"] == "output"]
        for in_node in global_input_nodes:
            for out_node in output_nodes:
                innovation_number += 1
                self.brain_connections.append({
                    "innovation_number": innovation_number, "start": in_node["number"],
                    "end": out_node["number"], "enabled": True, "weight": (random.uniform(-2.0, 2.0),) * 3
                })

        self.weights_and_biases = []
        self.build_weights_and_biases_list()

    def detect_recursion(self, n1_num, n2_num, visited=None):
        if visited is None:
            visited = set()
        if n2_num in visited:
            return False
        visited.add(n2_num)
        for conn in self.brain_connections:
            if not conn["enabled"]:
                continue
            if conn["start"] == n2_num:
                if conn["end"] == n1_num:
                    return True
                if self.detect_recursion(n1_num, conn["end"], visited):
                    return True
        return False

    def add_new_connection_brain(self):
        global innovation_number
        for attempt in range(100):
            n1 = random.choice(self.nodes)
            n2 = random.choice(self.nodes)
            if n1["number"] == n2["number"]:
                continue
            if n1["role"] == "output" or n2["role"] == "input":
                continue
            already_exists = any(
                c["start"] == n1["number"] and c["end"] == n2["number"] for c in self.brain_connections
            )
            if already_exists:
                continue
            if self.detect_recursion(n1["number"], n2["number"]):
                continue
            innovation_number += 1
            self.brain_connections.append({
                "innovation_number": innovation_number, "start": n1["number"],
                "end": n2["number"], "enabled": True, "weight": (random.uniform(-2.0, 2.0),) * 3
            })
            break

    def add_new_node_brain(self):
        global innovation_number
        enabled_connections = [c for c in self.brain_connections if c["enabled"]]
        if not enabled_connections:
            return
        split_conn = random.choice(enabled_connections)
        split_conn["enabled"] = False

        node_number = self.nodes[-1]["number"] + 1

        #the new hidden node gets its OWN innovation number (node_innovation),
        #separate from the connection-gene counter it also happens to share
        #-- this is what gives hidden nodes stable cross-genome identity for
        #crossover (see _node_key in main.py), the same role limb_gene_path
        #plays for in/out nodes
        innovation_number += 1
        new_node_innovation = innovation_number
        self.nodes.append({
            "number": node_number, "activation": random.choice(neuron_types), "bias": (0, 0, 0),
            "edge": f"hidden_{node_number}", "role": "hidden", "kind": "hidden", "depth": None,
            "limb_gene_path": None, "node_innovation": new_node_innovation,
            "last_output": (0, 0, 0)
        })

        innovation_number += 1
        self.brain_connections.append({
            "innovation_number": innovation_number, "start": split_conn["start"],
            "end": node_number, "enabled": True, "weight": (1.0, 1.0, 1.0)
        })
        innovation_number += 1
        self.brain_connections.append({
            "innovation_number": innovation_number, "start": node_number,
            "end": split_conn["end"], "enabled": True, "weight": split_conn["weight"]
        })

    #------------- CLONE -------------
    def clone(self, new_torso):
        #remaps limb-bound nodes' "edge" onto new_torso's own limb uuids,
        #matched by limb_gene_path (the stable structural identity) since
        #a fresh limb object always gets a fresh uuid. for a pure body
        #clone (identical gene_paths) every node should find a match; the
        #None-check exists for correctness if this is ever reused where
        #new_torso's body differs from self.torso's.
        new_limbs = new_torso.get_all_limbs()
        gene_path_to_new_name = {l.gene_path: l.name for l in new_limbs}

        new_brain = brain.__new__(brain)
        new_brain.name = str(uuid.uuid4())
        new_brain.torso = new_torso
        new_brain.torso_id = new_torso.name
        new_brain.input_names = list(self.input_names)
        new_brain.all_limbs = new_limbs

        new_brain.nodes = []
        new_brain.limb_output_numbers = {}
        for node in self.nodes:
            new_node = dict(node)
            if new_node["limb_gene_path"] is not None:
                new_edge = gene_path_to_new_name.get(new_node["limb_gene_path"])
                if new_edge is None:
                    continue  #structural mismatch -- limb doesn't exist on new_torso
                new_node["edge"] = new_edge
                if new_node["role"] == "output":
                    new_brain.limb_output_numbers[new_edge] = new_node["number"]
            new_brain.nodes.append(new_node)

        kept_numbers = {n["number"] for n in new_brain.nodes}
        new_brain.brain_connections = [
            dict(c) for c in self.brain_connections
            if c["start"] in kept_numbers and c["end"] in kept_numbers
        ]

        new_brain.weights_and_biases = []
        new_brain.build_weights_and_biases_list()
        return new_brain

    #------------- MUTATE -------------
    def mutate(self, mutation_rate, weight_perturb_std=0.5):
        for conn in self.brain_connections:
            if random.random() < mutation_rate:
                conn["weight"] = tuple(w + random.gauss(0, weight_perturb_std) for w in conn["weight"])
        for node in self.nodes:
            if random.random() < mutation_rate:
                node["bias"] = tuple(b + random.gauss(0, weight_perturb_std) for b in node["bias"])
            if random.random() < mutation_rate * 0.2:
                #swapping activation type is a bigger structural jump than
                #nudging a number, so it happens noticeably less often
                node["activation"] = random.choice(neuron_types)
        #structural mutations, rarer again, matching typical NEAT ratios
        if random.random() < mutation_rate * 0.25:
            self.add_new_connection_brain()
        if random.random() < mutation_rate * 0.1:
            self.add_new_node_brain()
        self.build_weights_and_biases_list()

    #------------- WEIGHTS & BIASES / HEBBIAN LEARNING -------------
    def build_weights_and_biases_list(self):
        #flat registry of every mutable weight/bias in this brain. each
        #entry holds a direct reference back to the connection/node dict
        #it came from, plus which key to write into -- this is what lets
        #apply_weights_and_biases push updates back into the real genome
        #without re-deriving where each value lives every time.
        self.weights_and_biases = []
        for conn in self.brain_connections:
            self.weights_and_biases.append({
                "kind": "weight", "ref": conn, "key": "weight", "value": conn["weight"]
            })
        for node in self.nodes:
            self.weights_and_biases.append({
                "kind": "bias", "ref": node, "key": "bias", "value": node["bias"]
            })
        return self.weights_and_biases

    def hebbian_update_weights_list(self, learning_rate=0.01, decay=0.001):
        #RMHL, per the design doc: ΔW = (LearningRate x R x Input x Output)
        #- Decay, applied elementwise across x/y/z. computed into each
        #entry's "value" field only -- brain_connections/nodes are NOT
        #touched here, so every update in one pass reads the SAME
        #pre-update state. call apply_weights_and_biases afterward to commit.
        #
        #CAVEAT: Decay is an unconditional subtraction, exactly as specified
        #in the design doc's formula -- applies even when Input x Output is
        #zero, which drives weights toward -inf rather than toward 0 over
        #many episodes with no counteracting signal. a proportional decay
        #(-decay * old_value) would be self-limiting instead -- implemented
        #literally as specified for now, not silently changed.
        #
        #ALSO: until evaluate_nn is implemented and populates "last_output"
        #from a real forward pass, every node's last_output defaults to
        #(0,0,0) -- so right now this only ever applies pure decay.
        R = self.torso.data["dopamine"]
        nodes_by_number = {n["number"]: n for n in self.nodes}

        for entry in self.weights_and_biases:
            if entry["kind"] == "weight":
                conn = entry["ref"]
                input_vec = nodes_by_number[conn["start"]].get("last_output", (0, 0, 0))
                output_vec = nodes_by_number[conn["end"]].get("last_output", (0, 0, 0))
                old = entry["value"]
                entry["value"] = tuple(
                    old[i] + (learning_rate * R * input_vec[i] * output_vec[i]) - decay
                    for i in range(3)
                )
            else:  #bias
                node = entry["ref"]
                output_vec = node.get("last_output", (0, 0, 0))
                old = entry["value"]
                #bias has no presynaptic "input" the way a connection does
                #-- modulated by the node's own output alone, same spirit
                #as a connection fed by a constant "1" input
                entry["value"] = tuple(
                    old[i] + (learning_rate * R * output_vec[i]) - decay
                    for i in range(3)
                )
        return self.weights_and_biases

    def apply_weights_and_biases(self):
        #iterates the flat list and writes each entry's value back into
        #the actual connection/node dict it came from
        for entry in self.weights_and_biases:
            entry["ref"][entry["key"]] = entry["value"]

    def evaluate_nn(self):
        pass
