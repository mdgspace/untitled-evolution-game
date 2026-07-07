import random
import math
from random import randrange
from collections import namedtuple
import uuid

#alternate neural schema to consider:
#use a CPPN/ HyperNEAT type architecture and have only one neural system, instead of 2 (global, local)
#so initally, a creature will evolve a brain such that each input is connected to a neuron, and they are directly connected to an output
#neuron at each limb. again, every input and output will be a vector3
#this CPPN will evolve using standard NEAT methods
#in this approach, we can mutate the creature's body as well (but very slowly)
#when we add a new limb, the existing connections and weights do not get affected!!


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
#the entire brain will deal with 3d vectors only
#i am treating 3d vectors as tuples with 3 elements in this python mock-up.
#later when translating it to game engine code, i'll use the engine's vector3 datatype instead
#some things are not 3d vectors, like frame number
#frame number will be encoded as (sin(2pi t/T), cos(that), t)
#any other input that's a single value n, can just be encoded as (n,n,n)

def sigmoid(vector, bias):
    return (1/(1 + math.exp(-vector[0]-bias[0])), 1/(1 + math.exp(-vector[1]-bias[1])), 1/(1 + math.exp(-vector[2]-bias[2])))

def sine(vector, bias):
    return (math.sin(vector[0]+bias[0]), math.sin(vector[1]+bias[1]), math.sin(vector[2]+bias[2]))

def relu(vector, bias):
    return (max(vector[0]+bias[0], 0), max(vector[1]+bias[1], 0), max(vector[2]+bias[2], 0))

def gaussian(vector, bias):
    return (math.exp(-((vector[0]+bias[0])**2 + (vector[1]+bias[1])**2 + (vector[2]+bias[2])**2)),)*3

neuron_types = [sigmoid, sine, relu, gaussian]


def place_limb(parent_dim, own_dim, existing, max_attempts=30):
    #treats parent and limb as bounding spheres (radius = half the largest dimension)
    #picks a random direction, places the limb just outside the parent's surface,
    #and rejects the position if it overlaps any sibling already placed on this parent
    parent_radius = max(parent_dim) / 2
    own_radius = max(own_dim) / 2
    distance = parent_radius + own_radius
    pos = [0, 0, 0]
    for i in range(max_attempts):
        vec = [random.uniform(-1, 1) for i in range(3)]
        length = math.sqrt(sum(v * v for v in vec)) or 1e-6
        direction = [v / length for v in vec]
        pos = [direction[i] * distance for i in range(3)]
        if all(math.dist(pos, other_pos) >= (own_radius + other_radius)
               for other_pos, other_radius in existing):
            return pos, own_radius
    return pos, own_radius  #fallback if no non-overlapping spot found in time


class torso():
    def get_new_connection(self):
        #generates d1 limb connections. helper function called in init
        dim = [randrange(2, 6) for i in range(3)]
        pos, radius = place_limb(self.dimensions, dim, self.placed_limb_positions)
        self.placed_limb_positions.append((pos, radius))
        ori = ["this is a random 3-d vector"]
        joint_type = random.choice(joint_types)
        return {
                "position": pos,
                "dimensions": dim,
                "orientation": ori,
                "joint_type": joint_type,
                "limb_object": limb(pos, dim, ori, joint_type)
               }

    def get_all_limbs(self):
        #walks every d1 connection's subtree and returns one flat list of every
        #limb in the creature, at any depth. this is what lets the brain wire
        #up an output neuron for every limb, not just the ones on the torso.
        all_limbs = []
        for conn in self.connections:
            all_limbs.extend(conn["limb_object"].get_subtree_limbs())
        return all_limbs

    #BRAIN RELATED FUNCTIONS
    def initialize_neat_brain(self):
        #initializes neat based brain. helper function called in init
        #the edge field indicates that it is an input or output layer neuron. so it can be connected to torso (input) or a limb (output)
        self.all_limbs = self.get_all_limbs()

        self.nodes = [{"number": 1, "activation": random.choice(neuron_types), "bias": (0, 0, 0), "edge": "torso", "depth": 0}]
        for i, limb_obj in enumerate(self.all_limbs):
            self.nodes.append({
                "number": i + 2,
                "activation": random.choice(neuron_types),
                "bias": (0, 0, 0),
                "edge": limb_obj.name,
                "depth": limb_obj.depth
            })

        self.brain_connections = [
            {"start": 1, "end": node["number"], "enabled": True, "weight": (random.uniform(-2.0, 2.0),) * 3}
            for node in self.nodes[1:]
        ]

    def add_new_connection_brain(self):
        #this shud make sure there are no recurrent connections as im not implementing memory of previous activations
        pass

    def add_new_node(self):
        pass

    def __init__(self):
        #this is called when we create a new creature (as one creature can only have one torso)
        self.mesh_var = "address to an object that has a cuboid mesh with collider"
        #mesh_var is basically what we will instantiate using the below parameters
        self.min_size = 10
        self.max_size = 20
        self.num_connections = randrange(1, 8)
        self.dimensions = [randrange(self.min_size, self.max_size) for i in range(3)]
        self.placed_limb_positions = []
        self.torso = {
            "mesh": self.mesh_var,
            "dimensions": self.dimensions
        }

        self.connections = [self.get_new_connection() for i in range(self.num_connections)]

        self.data = {
            "current_position": (0, 0, 0),
            #position of center of mass (origin)
            "position_memory": [],
            #at any point of time, this array has the positions of last 5 frames
            "energy": 100.0,
            #energy is the main fitness function (used after each generation). moving means energy down, eating means energy up etc
            "energy_memory": [],
            "dopamine": 0.0
            #this is the immediate fitness function. every frame it updates based on position, energy and their memories
            #it penalizes the creature for sitting still or for using energy in very large bursts
            #so that we get an optimum where it moves gradually in non-jerky movements
            #energy is the fitness used to update bodies and brain schemas
            #dopamine affects the weights and biases in the brain, as an iterative process
        }
        self.initialize_neat_brain()

    def clone():
        #passing .clone() on an instance should create an identical instance
        pass

    def mutate(self, mutation_rate):
        #SAVE THE EXISTING SCHEMA IN A FILE AND THEN MUTATE
        pass


class limb():
    max_depth = 5

    def get_new_connection(self):
        #this has same name as the func in torso, and looks kinda similar. but it is a different function
        dim = [randrange(2, 6) for i in range(3)]
        pos, radius = place_limb(self.dim, dim, [])
        ori = ["this is a random 3-d vector"]
        joint_type = random.choice(joint_types)
        return {
                "position": pos,
                "dimensions": dim,
                "orientation": ori,
                "joint_type": joint_type,
                "limb_object": limb(pos, dim, ori, joint_type, depth=self.depth + 1)
               }

    def get_subtree_limbs(self):
        #returns this limb plus every limb further down its own chain
        limbs = [self]
        if self.connection is not None:
            limbs.extend(self.connection["limb_object"].get_subtree_limbs())
        return limbs

    def __init__(self, pos, dim, ori, joint_type, depth=1):
        self.name = str(uuid.uuid4())
        self.dim = dim
        self.depth = depth
        self.has_connection = random.random() < 0.5 and self.depth < self.max_depth
        self.joint_type = joint_type
        self.limb = {"pos": pos,
                     "dim": dim,
                     "ori": ori
                     }

        self.connection = self.get_new_connection() if self.has_connection else None


def print_body(creature):
    print(f"TORSO  dim={creature.torso['dimensions']}")
    for conn in creature.connections:
        print_limb_subtree(conn, indent=1)

def print_limb_subtree(conn, indent):
    limb_obj = conn["limb_object"]
    joint_name = conn["joint_type"].__name__
    short_id = limb_obj.name[:8]
    prefix = "  " * indent + "└─ "
    print(f"{prefix}LIMB(d{limb_obj.depth})  id={short_id}  joint={joint_name}  dim={limb_obj.dim}")
    if limb_obj.connection is not None:
        print_limb_subtree(limb_obj.connection, indent + 1)


def print_brain(creature):
    print("BRAIN")
    nodes_by_number = {n["number"]: n for n in creature.nodes}
    input_node = nodes_by_number[1]
    print(f"  INPUT  node#{input_node['number']}  activation={input_node['activation'].__name__}  edge={input_node['edge']}")

    #sorted by depth so the printout reads top-down, roughly matching the body's own shape
    outgoing = sorted(creature.brain_connections, key=lambda c: nodes_by_number[c["end"]]["depth"])
    for conn in outgoing:
        target = nodes_by_number[conn["end"]]
        w = tuple(round(x, 2) for x in conn["weight"])
        limb_id = target["edge"][:8]  #shortened uuid, just for readability
        print(f"    ├─ weight={w} ─> OUTPUT node#{target['number']}  "
              f"limb={limb_id}  depth={target['depth']}  "
              f"activation={target['activation'].__name__}  bias={target['bias']}")


for i in range(5):
    creature = torso()
    print("CREATURE NUMBER ", i+1)
    print_body(creature)
    print_brain(creature)
    print("_______________________________________")
    print("_______________________________________\n")