import random
import math
import uuid

from torso import torso
from brain import brain as brain_class  #only needed by build_brain_from_genes,
                                         #which has to construct a brain instance
                                         #directly (bypassing __init__) for a
                                         #crossover child. everything else in
                                         #this file reaches limb/brain objects
                                         #by walking a torso instance, never by
                                         #importing limb.py at all.


#================= COMPATIBILITY DISTANCES =================

def body_compatibility_distance(creature_a, creature_b, c1=1.0, c2=1.0, c3=1.0, c4=0.5):
    #direct structural analogue of NEAT's brain compatibility formula --
    #match/disjoint/excess over body_innovation numbers instead of brain
    #innovation numbers, plus a "parameter difference" term (dimensions +
    #joint type) over the matching genes, playing the same role NEAT's
    #average weight difference plays for brains.
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
        self.representative = representative  #a creature (torso instance)
        self.members = []

class BodySpecies:
    def __init__(self, representative):
        self.representative = representative
        self.sub_species = []


def speciate_population(population, body_species_list=None):
    #body-species first (protects a novel LIMB/JOINT structure -- the
    #bigger evolutionary gamble), brain-sub-species second, only compared
    #against creatures that already share a body plan. representatives
    #persist across generations if a body_species_list is passed back in;
    #members always get rebuilt fresh each call.
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
    #shares fitness twice: within a brain sub-species first (protects
    #neural novelty among creatures with the same body plan), then across
    #the body species' total population (protects morphological novelty).
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
    #body before being kept.
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
    new_brain = brain_class.__new__(brain_class)
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


#================= DEMO =================

if __name__ == "__main__":
    population = [torso() for i in range(20)]

    print("=== SAMPLE CREATURE ===")
    print_body(population[0])
    print_brain(population[0])
    print()

    species_list = speciate_population(population)

    print("=== SPECIATION RESULT ===")
    for i, sp in enumerate(species_list):
        total = sum(len(sub.members) for sub in sp.sub_species)
        print(f"BodySpecies {i}  members={total}  sub_species={len(sp.sub_species)}")
        for j, sub in enumerate(sp.sub_species):
            print(f"    BrainSubSpecies {j}  members={len(sub.members)}")

    fitness_lookup = {c: random.uniform(0, 100) for c in population}
    shared = apply_fitness_sharing(species_list, fitness_lookup)
    print("\n=== SHARED FITNESS (first 5) ===")
    for c in population[:5]:
        print(f"  {c.name[:8]}  raw={fitness_lookup[c]:.1f}  shared={shared.get(c, 0):.4f}")

    print("\n=== CLONE-WITH-BRAIN-VARIATION ===")
    clones = spawn_body_clones_with_brain_variation(population[0], 3, brain_mutation_rate=0.4)
    for i, c in enumerate(clones):
        print(f"  clone {i}: brain_nodes={len(c.brain.nodes)}  brain_conns={len(c.brain.brain_connections)}")

    print("\n=== SEXUAL REPRODUCTION (brain crossover) ===")
    child = reproduce_sexually(population[0], fitness_lookup[population[0]],
                                population[1], fitness_lookup[population[1]])
    print_brain(child)
