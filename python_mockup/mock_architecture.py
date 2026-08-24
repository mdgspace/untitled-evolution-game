import random
import math
from random import randrange
import uuid

#for NEAT brains
innovation_number = 0


class joint():
    def __init__(self):
        pass

class ball_socket(joint):
    def __init__(self):
        pass

class hinge(joint):
    def __init__(self):
        pass

joint_types = [ball_socket, hinge]


#--- brain math nodes ---
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


#================= BODY SLOT SYSTEM =================

def get_available_slots(parent_dimensions, occupied_slot_ids):
    FACE_DIRECTIONS = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
    CORNER_DIRECTIONS = [(x, y, z) for x in (1, -1) for y in (1, -1) for z in (1, -1)]
    all_directions = FACE_DIRECTIONS + CORNER_DIRECTIONS
    slots = []
    for slot_id, direction in enumerate(all_directions):
        if slot_id in occupied_slot_ids:
            continue
        length = math.sqrt(sum(d * d for d in direction))
        unit = tuple(d / length for d in direction)
        slots.append({"slot_id": slot_id, "direction": unit})
    return slots


def place_limb(parent_dimensions, own_dim, occupied_slot_ids):
    slots = get_available_slots(parent_dimensions, occupied_slot_ids)
    if not slots:
        return None
    slot = random.choice(slots)
    parent_radius = max(parent_dimensions) / 2
    own_radius = max(own_dim) / 2
    distance = parent_radius + own_radius
    pos = tuple(d * distance for d in slot["direction"])
    return pos, own_radius, slot["slot_id"]


#================= BODY INNOVATION NUMBERS =================

body_innovation_number = 0
body_innovation_cache = {}

def get_body_innovation(gene_path):
    global body_innovation_number
    if gene_path in body_innovation_cache:
        return body_innovation_cache[gene_path]
    body_innovation_number += 1
    body_innovation_cache[gene_path] = body_innovation_number
    return body_innovation_number


class torso():
    def get_new_connection(self):
        dim = [randrange(2, 6) for i in range(3)]
        placement = place_limb(self.dimensions, dim, self.occupied_slots)
        if placement is None:
            return None
        pos, radius, slot_id = placement
        self.occupied_slots.add(slot_id)
        gene_path = (slot_id,)
        ori = ["this is a random 3-d vector"]
        joint_type = random.choice(joint_types)
        return {
                "position": pos,
                "dimensions": dim,
                "orientation": ori,
                "joint_type": joint_type,
                "slot_id": slot_id,
                "gene_path": gene_path,
                "body_innovation": get_body_innovation(gene_path),
                "limb_object": limb(pos, dim, ori, joint_type, gene_path)
               }

    def get_all_limbs(self):
        all_limbs = []
        for conn in self.connections:
            all_limbs.extend(conn["limb_object"].get_subtree_limbs())
        return all_limbs

    def get_all_body_genes(self):
        genes = []
        def walk(conn):
            genes.append(conn)
            child = conn["limb_object"]
            if child.connection is not None:
                walk(child.connection)
        for conn in self.connections:
            walk(conn)
        return genes

    def __init__(self):
        self.name = str(uuid.uuid4())
        self.mesh_var = "address to an object that has a cuboid mesh with collider"
        self.min_size = 10
        self.max_size = 20
        self.num_connections = randrange(1, 8)
        self.dimensions = [randrange(self.min_size, self.max_size) for i in range(3)]
        self.occupied_slots = set()
        self.torso = {
            "mesh": self.mesh_var,
            "dimensions": self.dimensions
        }
        self.connections = []
        for i in range(self.num_connections):
            conn = self.get_new_connection()
            if conn is not None:
                self.connections.append(conn)
        self.data = {
            "current_position": (0, 0, 0),
            "position_memory": [],
            "energy": 100.0,
            "energy_memory": [],
            "dopamine": 0.0
        }
        self.brain = brain(self)
        self.brain_id = self.brain.name

    #------------- CLONE -------------
    def clone(self):
        #produces a structurally IDENTICAL body (same dimensions, same
        #slots/gene_paths, same joint types) but with entirely new limb
        #and torso uuids, and a fresh (unmutated) brain remapped onto the
        #new limb identities. episodic data (position/energy/dopamine
        #history) resets -- a clone starts its own life, it doesn't
        #inherit the parent's in-progress episode.
        new_torso = torso.__new__(torso)
        new_torso.name = str(uuid.uuid4())
        new_torso.mesh_var = self.mesh_var
        new_torso.min_size = self.min_size
        new_torso.max_size = self.max_size
        new_torso.num_connections = self.num_connections
        new_torso.dimensions = list(self.dimensions)
        new_torso.occupied_slots = set(self.occupied_slots)
        new_torso.torso = {"mesh": new_torso.mesh_var, "dimensions": new_torso.dimensions}

        new_torso.connections = []
        for conn in self.connections:
            new_conn = dict(conn)
            new_limb_obj = clone_limb_subtree(conn["limb_object"])
            new_conn["limb_object"] = new_limb_obj
            #re-alias "dimensions" to the CLONE's own dim list, mirroring
            #the original aliasing between conn["dimensions"]/limb.dim/
            #limb.limb["dim"] -- see clone_limb_subtree for why this matters
            new_conn["dimensions"] = new_limb_obj.dim
            new_torso.connections.append(new_conn)

        new_torso.data = {
            "current_position": (0, 0, 0),
            "position_memory": [],
            "energy": 100.0,
            "energy_memory": [],
            "dopamine": 0.0
        }

        new_torso.brain = self.brain.clone(new_torso)
        new_torso.brain_id = new_torso.brain.name
        return new_torso

    #------------- MUTATE (body) -------------
    def mutate(self, mutation_rate):
        #perturbs existing dimensions/joint types. does NOT add or remove
        #limbs -- a topology change here would need matching output/input
        #node add/removal on the brain side too, which is the same gap
        #already flagged in add_new_node_brain's own comment. left as a
        #stub for that reason, not an oversight.
        if random.random() < mutation_rate:
            self.dimensions[:] = [randrange(self.min_size, self.max_size) for _ in range(3)]

        for limb_obj in self.get_all_limbs():
            if random.random() < mutation_rate:
                #in-place slice assignment, NOT reassignment -- limb.dim,
                #limb.limb["dim"], and this limb's parent connection's
                #"dimensions" are three aliases to the SAME list (see
                #get_new_connection). rebinding with "=" instead of "[:]="
                #would desync them; mutating in place keeps all three
                #pointing at the updated values.
                limb_obj.dim[:] = [randrange(2, 6) for _ in range(3)]
            if random.random() < mutation_rate:
                limb_obj.joint_type = random.choice(joint_types)
        #NOTE: position/orientation are not recomputed after a dimension
        #change, even though position technically depends on own_radius
        #(which depends on dimensions). a large resize could leave a
        #limb's stored position slightly inconsistent with its new size.
        #flagging as a known simplification -- recomputing would mean
        #walking the whole subtree, and this is the kind of thing better
        #handled by the actual geometry pass once this is in the engine.


def clone_limb_subtree(limb_obj):
    new_limb = limb.__new__(limb)
    new_limb.name = str(uuid.uuid4())
    new_dim = list(limb_obj.dim)  #new list, independent of the original
    new_limb.dim = new_dim
    new_limb.depth = limb_obj.depth
    new_limb.gene_path = limb_obj.gene_path
    new_limb.occupied_slots = set(limb_obj.occupied_slots)
    new_limb.has_connection = limb_obj.has_connection
    new_limb.joint_type = limb_obj.joint_type
    #re-alias "dim" inside the limb dict to the SAME new_dim list, exactly
    #mirroring the original three-way aliasing pattern (dim / limb["dim"]
    #/ connection["dimensions"]) but scoped entirely to this clone
    new_limb.limb = {"pos": limb_obj.limb["pos"], "dim": new_dim, "ori": limb_obj.limb["ori"]}

    if limb_obj.connection is not None:
        child_conn = dict(limb_obj.connection)
        child_limb = clone_limb_subtree(limb_obj.connection["limb_object"])
        child_conn["limb_object"] = child_limb
        child_conn["dimensions"] = child_limb.dim
        new_limb.connection = child_conn
    else:
        new_limb.connection = None
    return new_limb


class brain():
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
        #crossover_brains, the same role limb_gene_path plays for in/out nodes
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

    #------------- CLONE (brain) -------------
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

    #------------- MUTATE (brain) -------------
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
        #(weight perturbation most common, new connections less common,
        #new nodes rarest of all)
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
        #pre-update state (no order-dependency within a pass). call
        #apply_weights_and_biases afterward to commit.
        #
        #CAVEAT: Decay here is an unconditional subtraction, exactly as
        #specified in the design doc's formula -- it applies even when
        #Input x Output is zero. Applied repeatedly with no counteracting
        #signal, this drives weights toward -inf rather than toward 0.
        #Worth deciding deliberately whether you want a proportional decay
        #instead (-decay * old_value, which pulls toward zero and is
        #self-limiting) before running this over many episodes --
        #implemented literally as specified for now, not silently changed.
        #
        #ALSO: until evaluate_nn is implemented and populates "last_output"
        #from a real forward pass, every node's last_output defaults to
        #(0,0,0) -- so right now this will only ever apply pure decay, not
        #the Hebbian term. Expected placeholder behavior, not a bug.
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
        #the actual connection/node dict it came from -- the step that
        #makes hebbian_update_weights_list's changes actually affect
        #evaluate_nn on the next forward pass
        for entry in self.weights_and_biases:
            entry["ref"][entry["key"]] = entry["value"]

    def evaluate_nn(self):
        pass


class limb():
    max_depth = 5

    def get_new_connection(self):
        dim = [randrange(2, 6) for i in range(3)]
        placement = place_limb(self.dim, dim, self.occupied_slots)
        if placement is None:
            return None
        pos, radius, slot_id = placement
        self.occupied_slots.add(slot_id)
        gene_path = self.gene_path + (slot_id,)
        ori = ["this is a random 3-d vector"]
        joint_type = random.choice(joint_types)
        return {
                "position": pos,
                "dimensions": dim,
                "orientation": ori,
                "joint_type": joint_type,
                "slot_id": slot_id,
                "gene_path": gene_path,
                "body_innovation": get_body_innovation(gene_path),
                "limb_object": limb(pos, dim, ori, joint_type, gene_path, depth=self.depth + 1)
               }

    def get_subtree_limbs(self):
        limbs = [self]
        if self.connection is not None:
            limbs.extend(self.connection["limb_object"].get_subtree_limbs())
        return limbs

    def __init__(self, pos, dim, ori, joint_type, gene_path, depth=1):
        self.name = str(uuid.uuid4())
        self.dim = dim
        self.depth = depth
        self.gene_path = gene_path
        self.occupied_slots = set()
        self.has_connection = random.random() < 0.5 and self.depth < self.max_depth
        self.joint_type = joint_type
        self.limb = {"pos": pos, "dim": dim, "ori": ori}
        self.connection = self.get_new_connection() if self.has_connection else None


#================= COMPATIBILITY DISTANCES =================

def body_compatibility_distance(creature_a, creature_b, c1=1.0, c2=1.0, c3=1.0, c4=0.5):
    genes_a = {g["body_innovation"]: g for g in creature_a.get_all_body_genes()}
    genes_b = {g["body_innovation"]: g for g in creature_b.get_all_body_genes()}
    all_innov = set(genes_a) | set(genes_b)
    lower_max = min(max(genes_a, default=0), max(genes_b, default=0))
    matching, disjoint, excess = 0, 0, 0
    dim_diffs = []
    joint_mismatches = 0
    for innov in all_innov:
        if innov in genes_a and innov in genes_b:
            matching += 1
            ga, gb = genes_a[innov], genes_b[innov]
            dim_diffs.append(math.dist(ga["dimensions"], gb["dimensions"]))
            if ga["joint_type"] != gb["joint_type"]:
                joint_mismatches += 1
        elif innov > lower_max:
            excess += 1
        else:
            disjoint += 1
    N = max(len(genes_a), len(genes_b), 1)
    dim_bar = sum(dim_diffs) / len(dim_diffs) if dim_diffs else 0.0
    joint_term = joint_mismatches / matching if matching else 0.0
    return c1 * (excess / N) + c2 * (disjoint / N) + c3 * dim_bar + c4 * joint_term


def brain_compatibility_distance(brain_a, brain_b, c1=1.0, c2=1.0, c3=0.4):
    genes_a = {c["innovation_number"]: c for c in brain_a.brain_connections}
    genes_b = {c["innovation_number"]: c for c in brain_b.brain_connections}
    all_innov = set(genes_a) | set(genes_b)
    lower_max = min(max(genes_a, default=0), max(genes_b, default=0))
    matching, disjoint, excess, weight_diffs = 0, 0, 0, []
    for innov in all_innov:
        if innov in genes_a and innov in genes_b:
            matching += 1
            weight_diffs.append(math.dist(genes_a[innov]["weight"], genes_b[innov]["weight"]))
        elif innov > lower_max:
            excess += 1
        else:
            disjoint += 1
    N = max(len(genes_a), len(genes_b), 1)
    w_bar = sum(weight_diffs) / len(weight_diffs) if weight_diffs else 0.0
    return c1 * (excess / N) + c2 * (disjoint / N) + c3 * w_bar


#================= NESTED SPECIATION =================

BODY_COMPAT_THRESHOLD = 3.0
BRAIN_COMPAT_THRESHOLD = 2.0

class BrainSubSpecies:
    def __init__(self, representative):
        self.representative = representative
        self.members = []

class BodySpecies:
    def __init__(self, representative):
        self.representative = representative
        self.sub_species = []


def speciate_population(population, body_species_list=None):
    if body_species_list is None:
        body_species_list = []
    for sp in body_species_list:
        for sub in sp.sub_species:
            sub.members = []
    for creature in population:
        placed_body_species = None
        for sp in body_species_list:
            if body_compatibility_distance(creature, sp.representative) < BODY_COMPAT_THRESHOLD:
                placed_body_species = sp
                break
        if placed_body_species is None:
            placed_body_species = BodySpecies(representative=creature)
            body_species_list.append(placed_body_species)
        placed_sub = None
        for sub in placed_body_species.sub_species:
            if brain_compatibility_distance(creature.brain, sub.representative.brain) < BRAIN_COMPAT_THRESHOLD:
                placed_sub = sub
                break
        if placed_sub is None:
            placed_sub = BrainSubSpecies(representative=creature)
            placed_body_species.sub_species.append(placed_sub)
        placed_sub.members.append(creature)
    for sp in body_species_list:
        sp.sub_species = [sub for sub in sp.sub_species if sub.members]
    body_species_list = [sp for sp in body_species_list if sp.sub_species]
    return body_species_list


def apply_fitness_sharing(body_species_list, fitness_lookup):
    shared_fitness = {}
    for body_sp in body_species_list:
        total_body_members = sum(len(sub.members) for sub in body_sp.sub_species)
        if total_body_members == 0:
            continue
        for sub in body_sp.sub_species:
            sub_size = len(sub.members)
            if sub_size == 0:
                continue
            for creature in sub.members:
                raw = fitness_lookup[creature]
                shared_fitness[creature] = raw / sub_size / total_body_members
    return shared_fitness


#================= BODY-CLONE-WITH-BRAIN-VARIATION =================

def spawn_body_clones_with_brain_variation(parent_torso, num_clones, brain_mutation_rate=0.1):
    #produces num_clones creatures with an IDENTICAL body to parent_torso,
    #but an independently mutated brain each -- clones diverge only in
    #wiring/behavior, never in body plan.
    clones = []
    for i in range(num_clones):
        child = parent_torso.clone()
        child.brain.mutate(brain_mutation_rate)
        clones.append(child)
    return clones


#================= BRAIN CROSSOVER =================

def _node_key(node):
    #stable cross-genome identity for a node, used to align genes between
    #two DIFFERENT creatures (whose raw node numbers and limb uuids will
    #generally differ even for structurally identical body plans):
    #  - hidden nodes: their own node_innovation number
    #  - global inputs: (kind, edge) -- edge is a stable name like "torso",
    #    not a per-instance uuid, so it's usable directly
    #  - limb-bound nodes: (kind, limb_gene_path) -- gene_path is the
    #    stable structural identity; the limb's actual uuid is NOT used
    #    here since two different creatures never share limb uuids even
    #    when their bodies are structurally identical
    if node["role"] == "hidden":
        return ("hidden", node["node_innovation"])
    if node["kind"] == "global":
        return ("global", node["edge"])
    return (node["kind"], node["limb_gene_path"])


def crossover_brains(brain_a, fitness_a, brain_b, fitness_b, child_torso):
    #standard NEAT crossover (matching genes: coin flip; disjoint/excess:
    #fitter parent only, or both if equal fitness), adapted so that node
    #alignment uses _node_key instead of raw node number, and every
    #inherited node/connection gets validated against child_torso's own
    #body before being kept -- this is the "crossover validation" step
    #flagged as necessary all the way back when we discussed Sims' own
    #approach requiring exactly this kind of repair pass.
    #
    #NOTE: child_torso's BODY is supplied by the caller, not produced by
    #this function -- body crossover isn't implemented yet, so in practice
    #the caller typically passes a clone of one parent's body (see
    #reproduce_sexually below). only the BRAIN is a genuine crossover of
    #both parents here.
    if fitness_a > fitness_b:
        fitter, equal_fitness = brain_a, False
    elif fitness_b > fitness_a:
        fitter, equal_fitness = brain_b, False
    else:
        fitter, equal_fitness = None, True

    nodes_a = {_node_key(n): n for n in brain_a.nodes}
    nodes_b = {_node_key(n): n for n in brain_b.nodes}
    number_to_key_a = {n["number"]: _node_key(n) for n in brain_a.nodes}
    number_to_key_b = {n["number"]: _node_key(n) for n in brain_b.nodes}

    child_gene_paths = {l.gene_path for l in child_torso.get_all_limbs()}

    def node_survives_on_child(node):
        if node["limb_gene_path"] is None:
            return True
        return node["limb_gene_path"] in child_gene_paths

    #--- select node genes ---
    all_keys = set(nodes_a) | set(nodes_b)
    selected_nodes = {}
    for key in all_keys:
        in_a, in_b = key in nodes_a, key in nodes_b
        if in_a and in_b:
            source = random.choice([nodes_a[key], nodes_b[key]])
        elif equal_fitness:
            source = nodes_a[key] if in_a else nodes_b[key]
        else:
            fitter_nodes = nodes_a if fitter is brain_a else nodes_b
            if key not in fitter_nodes:
                continue
            source = fitter_nodes[key]
        if not node_survives_on_child(source):
            continue
        selected_nodes[key] = source

    #--- renumber for the child, remap limb-bound edges onto child_torso ---
    gene_path_to_child_name = {l.gene_path: l.name for l in child_torso.get_all_limbs()}
    key_to_new_number = {}
    child_nodes = []
    next_number = 1
    for key, node in selected_nodes.items():
        new_node = dict(node)
        new_node["number"] = next_number
        if new_node["limb_gene_path"] is not None:
            new_node["edge"] = gene_path_to_child_name.get(new_node["limb_gene_path"], new_node["edge"])
        child_nodes.append(new_node)
        key_to_new_number[key] = next_number
        next_number += 1

    #--- select connection genes, remap start/end via node key ---
    conns_a = {c["innovation_number"]: c for c in brain_a.brain_connections}
    conns_b = {c["innovation_number"]: c for c in brain_b.brain_connections}
    all_innov = set(conns_a) | set(conns_b)

    child_connections = []
    for innov in all_innov:
        in_a, in_b = innov in conns_a, innov in conns_b
        if in_a and in_b:
            ga, gb = conns_a[innov], conns_b[innov]
            owner = random.choice(["a", "b"])
            source = dict(ga if owner == "a" else gb)
            if not ga["enabled"] or not gb["enabled"]:
                if random.random() < 0.75:  #classic NEAT convention
                    source["enabled"] = False
        elif equal_fitness:
            owner = "a" if in_a else "b"
            source = dict(conns_a[innov] if in_a else conns_b[innov])
        else:
            owner = "a" if fitter is brain_a else "b"
            fitter_conns = conns_a if fitter is brain_a else conns_b
            if innov not in fitter_conns:
                continue
            source = dict(fitter_conns[innov])

        number_to_key = number_to_key_a if owner == "a" else number_to_key_b
        start_key = number_to_key.get(source["start"])
        end_key = number_to_key.get(source["end"])
        if start_key not in key_to_new_number or end_key not in key_to_new_number:
            continue  #an endpoint didn't survive node selection/validation
        source["start"] = key_to_new_number[start_key]
        source["end"] = key_to_new_number[end_key]
        child_connections.append(source)

    return child_nodes, child_connections


def build_brain_from_genes(child_torso, nodes, connections):
    new_brain = brain.__new__(brain)
    new_brain.name = str(uuid.uuid4())
    new_brain.torso = child_torso
    new_brain.torso_id = child_torso.name
    new_brain.input_names = [n["edge"] for n in nodes if n["kind"] == "global"]
    new_brain.all_limbs = child_torso.get_all_limbs()
    new_brain.nodes = nodes
    new_brain.limb_output_numbers = {n["edge"]: n["number"] for n in nodes if n["role"] == "output"}
    new_brain.brain_connections = connections
    new_brain.weights_and_biases = []
    new_brain.build_weights_and_biases_list()
    return new_brain


def reproduce_sexually(torso_a, fitness_a, torso_b, fitness_b):
    #body crossover isn't implemented yet -- the child's BODY is a clone
    #of whichever parent is fitter (ties broken toward parent a). only the
    #BRAIN is a genuine crossover of both parents. real simplification,
    #not a hidden design decision.
    fitter_torso = torso_a if fitness_a >= fitness_b else torso_b
    child_torso = fitter_torso.clone()  #clone() builds a matching brain too -- discarded, real crossover replaces it below
    nodes, connections = crossover_brains(torso_a.brain, fitness_a, torso_b.brain, fitness_b, child_torso)
    child_torso.brain = build_brain_from_genes(child_torso, nodes, connections)
    child_torso.brain_id = child_torso.brain.name
    return child_torso


#================= PRINTING =================

def print_body(creature):
    print(f"TORSO  id={creature.name[:8]}  dim={creature.torso['dimensions']}")
    for conn in creature.connections:
        print_limb_subtree(conn, indent=1)

def print_limb_subtree(conn, indent):
    limb_obj = conn["limb_object"]
    joint_name = conn["joint_type"].__name__
    short_id = limb_obj.name[:8]
    prefix = "  " * indent + "|- "
    print(f"{prefix}LIMB(d{limb_obj.depth})  id={short_id}  slot={conn['slot_id']}  "
          f"path={conn['gene_path']}  body_innov={conn['body_innovation']}  joint={joint_name}  dim={limb_obj.dim}")
    if limb_obj.connection is not None:
        print_limb_subtree(limb_obj.connection, indent + 1)

def print_brain(creature):
    b = creature.brain
    print(f"BRAIN  id={b.name[:8]}  (torso id={b.torso_id[:8]})  nodes={len(b.nodes)}  connections={len(b.brain_connections)}")