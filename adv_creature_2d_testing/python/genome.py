"""Validated body genomes and innovation-aligned evolution primitives."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple
import copy
import math
import random
import uuid

MAX_DEPTH = 5
MAX_LIMBS = 8
SLOTS = 8


@dataclass
class LimbGene:
    innovation_id: int
    parent_innovation_id: int
    attachment_slot: int
    width: float = 1.5
    height: float = 0.45
    mass: float = 1.0
    inertia: float = 0.2
    joint_type: str = "hinge"
    min_angle: float = -90.0
    max_angle: float = 90.0
    max_torque: float = 50.0
    enabled: bool = True


@dataclass
class BodyGenome:
    genome_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    torso_width: float = 1.5
    torso_height: float = 1.5
    torso_mass: float = 2.0
    torso_inertia: float = 0.5
    limbs: List[LimbGene] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict) -> "BodyGenome":
        allowed = set(LimbGene.__dataclass_fields__)
        limbs = [LimbGene(**{k: v for k, v in raw.items() if k in allowed}) for raw in value.get("limbs", [])]
        return cls(
            genome_id=value.get("genome_id", str(uuid.uuid4())),
            torso_width=float(value.get("torso_width", 1.5)),
            torso_height=float(value.get("torso_height", 1.5)),
            torso_mass=float(value.get("torso_mass", 2.0)),
            torso_inertia=float(value.get("torso_inertia", 0.5)),
            limbs=limbs,
        ).validate()

    def clone(self) -> "BodyGenome":
        result = copy.deepcopy(self)
        result.genome_id = str(uuid.uuid4())
        return result

    def paths(self) -> Dict[int, Tuple[int, ...]]:
        genes = {g.innovation_id: g for g in self.limbs if g.enabled}
        result: Dict[int, Tuple[int, ...]] = {}

        def resolve(gene: LimbGene, seen: set[int]) -> Optional[Tuple[int, ...]]:
            if gene.innovation_id in seen:
                return None
            if gene.parent_innovation_id == 0:
                return (gene.attachment_slot,)
            parent = genes.get(gene.parent_innovation_id)
            if parent is None:
                return None
            parent_path = result.get(parent.innovation_id) or resolve(parent, seen | {gene.innovation_id})
            return None if parent_path is None else parent_path + (gene.attachment_slot,)

        for gene in genes.values():
            path = resolve(gene, set())
            if path is not None:
                result[gene.innovation_id] = path
        return result

    def validate(self) -> "BodyGenome":
        """Deterministically prune malformed, cyclic, duplicate, or unsupported structure."""
        self.torso_width = _clamp(self.torso_width, .5, 3.0)
        self.torso_height = _clamp(self.torso_height, .5, 3.0)
        self.torso_mass = _clamp(self.torso_mass, .1, 20.0)
        self.torso_inertia = _clamp(self.torso_inertia, .01, 20.0)
        remaining = sorted((copy.deepcopy(g) for g in self.limbs if g.enabled), key=lambda g: g.innovation_id)
        accepted: List[LimbGene] = []
        accepted_ids = {0}
        occupied = set()
        depth_by_id = {0: 0}
        positions = {0: (0.0, 0.0)}
        dimensions = {0: (self.torso_width, self.torso_height)}
        while remaining and len(accepted) < MAX_LIMBS:
            progressed = False
            for gene in list(remaining):
                if gene.parent_innovation_id not in accepted_ids:
                    continue
                remaining.remove(gene)
                progressed = True
                key = (gene.parent_innovation_id, int(gene.attachment_slot))
                depth = depth_by_id[gene.parent_innovation_id] + 1
                if key in occupied or not 0 <= gene.attachment_slot < SLOTS or depth > MAX_DEPTH:
                    continue
                if gene.joint_type != "hinge":
                    continue
                gene.width = _clamp(gene.width, .3, 3.0)
                gene.height = _clamp(gene.height, .15, 1.2)
                gene.mass = _clamp(gene.mass, .05, 10.0)
                gene.inertia = _clamp(gene.inertia, .005, 10.0)
                gene.min_angle = _clamp(gene.min_angle, -170.0, -1.0)
                gene.max_angle = _clamp(gene.max_angle, 1.0, 170.0)
                gene.max_torque = _clamp(gene.max_torque, 1.0, 200.0)
                angle = (math.pi / 2, -math.pi / 2, math.pi, 0.0,
                         math.pi / 4, -math.pi / 4, 3 * math.pi / 4, -3 * math.pi / 4)[gene.attachment_slot]
                direction = (math.cos(angle), math.sin(angle))
                parent_position = positions[gene.parent_innovation_id]
                parent_size = dimensions[gene.parent_innovation_id]
                distance = max(parent_size) / 2 + gene.width / 2
                position = (parent_position[0] + direction[0] * distance,
                            parent_position[1] + direction[1] * distance)
                obvious_intersection = any(
                    other_id != gene.parent_innovation_id
                    and math.dist(position, other_position) < (min(gene.width, gene.height) + min(dimensions[other_id])) * .25
                    for other_id, other_position in positions.items()
                )
                if obvious_intersection:
                    continue
                accepted.append(gene)
                accepted_ids.add(gene.innovation_id)
                occupied.add(key)
                depth_by_id[gene.innovation_id] = depth
                positions[gene.innovation_id] = position
                dimensions[gene.innovation_id] = (gene.width, gene.height)
            if not progressed:
                break
        self.limbs = accepted
        return self


class InnovationRegistry:
    """Global structural history: identical mutation signatures reuse innovations."""
    def __init__(self, next_body_innovation: int = 1, structural: Optional[Dict[str, int]] = None):
        self.next_body_innovation = int(next_body_innovation)
        self.structural = dict(structural or {})

    def body_innovation(self, parent: int = 0, slot: int = 0, operation: str = "add") -> int:
        key = f"{operation}:{int(parent)}:{int(slot)}"
        if key not in self.structural:
            self.structural[key] = self.next_body_innovation
            self.next_body_innovation += 1
        return self.structural[key]

    def to_dict(self) -> Dict:
        return {"next_body_innovation": self.next_body_innovation, "structural": dict(self.structural)}

    @classmethod
    def from_dict(cls, value: Dict) -> "InnovationRegistry":
        return cls(value.get("next_body_innovation", 1), value.get("structural", {}))


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, float(value)))


def _add(
    registry: InnovationRegistry,
    parent: int,
    slot: int,
    operation: str = "add",
    **kwargs,
) -> LimbGene:
    return LimbGene(
        registry.body_innovation(parent, slot, operation),
        parent,
        slot,
        **kwargs,
    )


def seed_templates(registry: InnovationRegistry) -> List[BodyGenome]:
    """Three deterministic seed morphologies, with shared IDs for shared structure."""
    common = [_add(registry, 0, slot) for slot in (0, 2, 4, 6)]
    first = BodyGenome(limbs=copy.deepcopy(common)).validate()
    second_genes = copy.deepcopy(common)
    second_genes += [_add(registry, 0, 1, width=1.1), _add(registry, 0, 7, width=1.1)]
    second = BodyGenome(torso_width=1.8, torso_height=1.2, limbs=second_genes).validate()
    third_genes = copy.deepcopy(common[:2])
    third_genes += [_add(registry, common[0].innovation_id, 0, width=1.0),
                    _add(registry, common[1].innovation_id, 4, width=1.0)]
    third = BodyGenome(torso_width=1.2, torso_height=1.8, limbs=third_genes).validate()
    return [first, second, third]


def default_body(registry: InnovationRegistry) -> BodyGenome:
    return seed_templates(registry)[0].clone()


def body_distance(a: BodyGenome, b: BodyGenome) -> float:
    aa = {g.innovation_id: g for g in a.limbs}
    bb = {g.innovation_id: g for g in b.limbs}
    keys = set(aa) | set(bb)
    if not keys:
        return abs(a.torso_width - b.torso_width) + abs(a.torso_height - b.torso_height)
    structural = sum(key not in aa or key not in bb for key in keys) / len(keys)
    matching = set(aa) & set(bb)
    param = 0.0 if not matching else sum(
        abs(aa[k].width - bb[k].width) / 3.0 + abs(aa[k].height - bb[k].height) / 1.2
        + abs(aa[k].max_torque - bb[k].max_torque) / 200.0 for k in matching
    ) / len(matching)
    torso = (abs(a.torso_width - b.torso_width) + abs(a.torso_height - b.torso_height)) / 6.0
    return structural + .35 * param + .25 * torso


def _descendants(genes: Iterable[LimbGene], root: int) -> set[int]:
    result = {root}
    changed = True
    while changed:
        changed = False
        for gene in genes:
            if gene.parent_innovation_id in result and gene.innovation_id not in result:
                result.add(gene.innovation_id)
                changed = True
    return result


def crossover_body(a: BodyGenome, fitness_a: float, b: BodyGenome, fitness_b: float,
                   registry: InnovationRegistry) -> BodyGenome:
    fitter, donor_parent = (a, b) if fitness_a <= fitness_b else (b, a)
    child = fitter.clone()
    donor_by_id = {g.innovation_id: g for g in donor_parent.limbs}
    for gene in child.limbs:
        donor = donor_by_id.get(gene.innovation_id)
        if donor and random.random() < .5:
            for name in ("width", "height", "mass", "inertia", "min_angle", "max_angle", "max_torque"):
                setattr(gene, name, getattr(donor, name))
    child_ids = {0} | {g.innovation_id for g in child.limbs}
    occupied = {(g.parent_innovation_id, g.attachment_slot) for g in child.limbs}
    roots = [g for g in donor_parent.limbs if g.innovation_id not in child_ids
             and g.parent_innovation_id in child_ids and (g.parent_innovation_id, g.attachment_slot) not in occupied]
    if roots:
        ids = _descendants(donor_parent.limbs, random.choice(roots).innovation_id)
        child.limbs.extend(copy.deepcopy(g) for g in donor_parent.limbs if g.innovation_id in ids)
    return child.validate()


def mutate_body(body: BodyGenome, registry: InnovationRegistry) -> BodyGenome:
    child = body.clone()
    child.torso_width += random.gauss(0, .08)
    child.torso_height += random.gauss(0, .08)
    for gene in child.limbs:
        if random.random() < .30:
            gene.width += random.gauss(0, .12)
            gene.height += random.gauss(0, .06)
            gene.mass += random.gauss(0, .08)
            gene.max_torque += random.gauss(0, 5.0)
    if random.random() < .20:
        mode = random.choice(("add", "remove", "replace"))
        if mode == "add" and len(child.limbs) < MAX_LIMBS:
            parent = random.choice([0] + [g.innovation_id for g in child.limbs])
            occupied = {(g.parent_innovation_id, g.attachment_slot) for g in child.limbs}
            slots = [slot for slot in range(SLOTS) if (parent, slot) not in occupied]
            if slots:
                child.limbs.append(_add(registry, parent, random.choice(slots)))
        elif mode in ("remove", "replace") and len(child.limbs) > 1:
            victim = random.choice(child.limbs)
            removed = _descendants(child.limbs, victim.innovation_id)
            child.limbs = [g for g in child.limbs if g.innovation_id not in removed]
            if mode == "replace":
                child.limbs.append(_add(registry, victim.parent_innovation_id, victim.attachment_slot,
                                        operation="replace", width=victim.width + random.gauss(0, .2),
                                        height=victim.height + random.gauss(0, .1)))
    return child.validate()
