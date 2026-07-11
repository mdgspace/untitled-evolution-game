import random
from random import randrange
import uuid

from limb import limb, place_limb, get_body_innovation, joint_types, clone_limb_subtree
from brain import brain


class torso():
    def get_new_connection(self):
        #generates one d1 limb connection, or None if the torso has no
        #free slots left
        dim = [randrange(2, 6) for i in range(3)]
        placement = place_limb(self.dimensions, dim, self.occupied_slots)
        if placement is None:
            return None
        pos, radius, slot_id = placement
        self.occupied_slots.add(slot_id)
        gene_path = (slot_id,)  #torso is always the implicit root, so a
                                 #d1 limb's whole identity is just its own slot
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
        #flat list of every connection dict in the creature (the body's
        #equivalent of NEAT's flat connection-gene list), used by
        #body_compatibility_distance in main.py
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
        #this is called when we create a new creature (as one creature can only have one torso)
        self.name = str(uuid.uuid4())
        self.mesh_var = "address to an object that has a cuboid mesh with collider"
        #mesh_var is basically what we will instantiate using the below parameters
        self.min_size = 10
        self.max_size = 20
        self.num_connections = randrange(1, 8)
        self.dimensions = [randrange(self.min_size, self.max_size) for i in range(3)]
        self.occupied_slots = set()
        self.torso = {
            "mesh": self.mesh_var,
            "dimensions": self.dimensions
        }

        #skip any attempt that returns None (parent out of slots) rather
        #than crashing -- means len(self.connections) can be less than
        #self.num_connections if slots run out
        self.connections = []
        for i in range(self.num_connections):
            conn = self.get_new_connection()
            if conn is not None:
                self.connections.append(conn)

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
        }

        #brain is its own object. torso and brain hold a direct reference
        #to each other AND each other's uuid, same pattern as a connection
        #dict holding a direct "limb_object" reference while the limb
        #itself carries its own identity in "name"
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
            #limb.limb["dim"] -- see clone_limb_subtree in limb.py for why
            #this matters
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

    #------------- MUTATE -------------
    def mutate(self, mutation_rate):
        #perturbs existing dimensions/joint types. does NOT add or remove
        #limbs -- a topology change here would need matching output/input
        #node add/removal on the brain side too, which is the same gap
        #already flagged in brain.add_new_node_brain's own comment. left
        #as a stub for that reason, not an oversight.
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
        #change -- flagged as a known simplification, see limb.py.
