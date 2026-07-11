import random
import math
from random import randrange
import uuid

#================= JOINTS =================

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


#================= BODY SLOT SYSTEM =================

def get_available_slots(parent_dimensions, occupied_slot_ids):
    #STUB -- will be replaced once ported to the game engine, where slots
    #should come from real geometry (valid attachment points on the actual
    #mesh/collider surface), not a hardcoded direction list. What has to
    #survive the rewrite: a FIXED, ENUMERABLE set of slot_ids per parent,
    #so the same slot_id reliably means "the same attachment point" across
    #every creature that has that parent shape. That stability is the
    #entire reason body innovation numbers can work below.
    #
    #for now: 6 face-center directions + 8 corner directions = 14 fixed
    #slots, same list for every parent regardless of its actual dimensions.
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
    #returns None if the parent has no room left, which callers must handle.
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
body_innovation_cache = {}  #gene_path (tuple of slot_ids from root) -> innovation number

def get_body_innovation(gene_path):
    #same principle as the brain's global innovation_number cache: if this
    #exact structural gene_path has been assigned a number before -- in
    #ANY creature, this generation or a past one -- reuse it. two
    #creatures independently evolving a limb at path (3,1) are recognized
    #as having the "same" mutation, which is what makes gene-by-gene
    #alignment in body_compatibility_distance meaningful (see main.py).
    global body_innovation_number
    if gene_path in body_innovation_cache:
        return body_innovation_cache[gene_path]
    body_innovation_number += 1
    body_innovation_cache[gene_path] = body_innovation_number
    return body_innovation_number


#================= LIMB =================

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
        #returns this limb plus every limb further down its own chain
        limbs = [self]
        if self.connection is not None:
            limbs.extend(self.connection["limb_object"].get_subtree_limbs())
        return limbs

    def __init__(self, pos, dim, ori, joint_type, gene_path, depth=1):
        self.name = str(uuid.uuid4())
        self.dim = dim
        self.depth = depth
        self.gene_path = gene_path      #this limb's own structural identity
        self.occupied_slots = set()     #slots used by ITS children
        self.has_connection = random.random() < 0.5 and self.depth < self.max_depth
        self.joint_type = joint_type
        self.limb = {"pos": pos, "dim": dim, "ori": ori}
        self.connection = self.get_new_connection() if self.has_connection else None


def clone_limb_subtree(limb_obj):
    #deep-clones a limb and everything further down its chain. fresh uuid
    #for every clone, but gene_path is preserved exactly -- gene_path is
    #the identity that actually matters structurally, not the uuid.
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
    #/ connection["dimensions"]) but scoped entirely to this clone --
    #see torso.clone() in torso.py, which does the matching re-alias for
    #the connection dict itself
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
