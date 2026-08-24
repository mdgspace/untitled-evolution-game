"""Python-authoritative, asynchronous, continuously learning ecosystem.

The network thread only snapshots/acts.  Learning consumes already executed
physics segments in the background, so a slow optimizer cannot freeze Unity.
"""
from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass, field
import math
import hashlib
import os
import queue
import random
import shutil
import threading
import time
import uuid
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import torch
import torch.optim as optim

from genome import BodyGenome, InnovationRegistry, body_distance, crossover_body, mutate_body, seed_templates
from live_brain import (DEVICE, HISTORY, MAX_DELTA_DEGREES, MAX_PATH_DEPTH, MODEL_KEYS, ROOT_DIM, STATE_DIM,
                        LiveCreatureBrain, SharedStochasticPolicy, TemporalAchiever, move_batch, optimizer_to,
                        pad_batches)
from neat_goal_setter import NEATGoalSetter, NEATInnovationRegistry, compatibility_distance as m1_distance

PROTOCOL_VERSION = 4
SCHEMA_VERSION = 5
POPULATION = 4
REPLACEMENTS = 1
COHORT_SECONDS = 60.0
MIN_PARENT_SAMPLES = 100
CHECKPOINT_SECONDS = 30.0
REPLAY_SIZE = 5000
M3_BATCH_SIZE = 8
MIN_GOAL = .05
M3_MOTION_THRESHOLD = .001


def _clone_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in batch.items()}


@dataclass
class PendingAction:
    batch: Dict[str, torch.Tensor]
    action: torch.Tensor
    log_prob: float
    value: float
    goal: torch.Tensor
    source_tick: int


@dataclass
class Transition:
    creature_id: str
    batch: Dict[str, torch.Tensor]
    action: torch.Tensor
    log_prob: float
    value: float
    displacement: torch.Tensor
    reward: float
    source_tick: int


@dataclass
class Champion:
    body: BodyGenome
    m1: Dict[str, Any]
    fitness: float
    energy_rate: float
    def to_dict(self) -> Dict[str, Any]: return {"body": self.body.to_dict(), "m1": deepcopy(self.m1), "fitness": self.fitness, "energy_rate": self.energy_rate}
    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "Champion": return cls(BodyGenome.from_dict(value["body"]), deepcopy(value["m1"]), float(value.get("fitness", float("inf"))), float(value.get("energy_rate", 0)))


@dataclass
class CreatureRecord:
    creature_id: str
    body: BodyGenome
    brain: LiveCreatureBrain  # M1 genotype; population-wide M2 lives on LiveEcosystem.
    species_id: str
    m1_species_id: str
    born_seconds: float
    history: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=HISTORY))
    behavior: Deque[float] = field(default_factory=lambda: deque(maxlen=300))
    energy_totals: Dict[str, float] = field(default_factory=lambda: {"food_gain": 0., "baseline_cost": 0., "movement_cost": 0., "idle_cost": 0.})
    last_energy_counters: Dict[str, float] = field(default_factory=dict)
    pending_actions: Dict[str, PendingAction] = field(default_factory=dict)
    last_root: Optional[Tuple[float, float]] = None
    latest_goal: Tuple[float, float] = (0., 0.)
    death_penalty: float = 0.
    control_steps: int = 0
    def age(self, now: float) -> float: return max(0., now - self.born_seconds)
    def body_fitness(self) -> float: return -sum(self.behavior) / len(self.behavior) if len(self.behavior) >= MIN_PARENT_SAMPLES else float("inf")
    def energy_rate(self, now: float) -> float:
        spent = self.energy_totals["baseline_cost"] + self.energy_totals["movement_cost"] + self.energy_totals["idle_cost"]
        return (self.energy_totals["food_gain"] - spent - self.death_penalty) / max(1., self.age(now))


class LiveEcosystem:
    def __init__(self, checkpoint_path: str):
        self.lock, self.model_lock = threading.RLock(), threading.RLock()
        self.checkpoint_path = checkpoint_path
        self.registry, self.m1_registry = InnovationRegistry(), NEATInnovationRegistry()
        self.creatures: Dict[str, CreatureRecord] = {}
        self.body_species_representatives: Dict[str, BodyGenome] = {}
        self.body_champions: Dict[str, Champion] = {}
        self.global_champion: Optional[Champion] = None
        self.body_threshold, self.m1_threshold = .25, .45
        self.clock, self.next_cohort = 0., COHORT_SECONDS
        self.policy = SharedStochasticPolicy()
        self.inference_policy = self.policy.clone_for_inference()
        self.m3 = TemporalAchiever(output_dim=2).to(DEVICE)
        self.m3_optimizer = optim.Adam(self.m3.parameters(), lr=.001)
        self.replay: Deque[Tuple[Dict[str, torch.Tensor], torch.Tensor]] = deque(maxlen=REPLAY_SIZE)
        self.transition_queue: "queue.SimpleQueue[Transition]" = queue.SimpleQueue()
        self.stop_event = threading.Event()
        self.learner = threading.Thread(target=self._learner_loop, name="LiveEcosystemLearner", daemon=True)
        self.learner_updates = self.snapshot_version = 0
        self.last_m3_error = self.last_observe_ms = self.last_inference_ms = self.last_learner_update_ms = 0.
        self.last_observe_tick = self.last_inference_tick = self.last_learner_tick = -1
        self.learner_busy = False
        self.dropped_learning = 0
        self.checkpoint_status, self.last_checkpoint = "Not saved", 0.
        self.pending_commands: Dict[str, Dict[str, Any]] = {}
        self.processed_event_ids, self._processed_events = deque(maxlen=4096), set()
        self.session_id = uuid.uuid4().hex
        self._load_or_seed()
        self._publish_snapshot()
        self.learner.start()

    # Persistence keeps replay CPU-portable and rejects incompatible controller schemas safely.
    def _load_or_seed(self) -> None:
        if os.path.exists(self.checkpoint_path):
            try:
                data = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
                if int(data.get("schema", 0)) != SCHEMA_VERSION: raise ValueError("checkpoint is not asynchronous schema v5")
                self.registry = InnovationRegistry.from_dict(data["registry"]); self.m1_registry = NEATInnovationRegistry.from_dict(data["m1_registry"])
                self.clock, self.next_cohort = float(data.get("clock", 0)), float(data.get("next_cohort", COHORT_SECONDS))
                self.body_threshold, self.m1_threshold = float(data.get("body_threshold", .25)), float(data.get("m1_threshold", .45))
                self.policy.load_state_dict(data["policy"]); self.m3.load_state_dict(data["m3"]); self.m3_optimizer.load_state_dict(data["m3_optimizer"]); optimizer_to(self.m3_optimizer, DEVICE)
                self.snapshot_version, self.learner_updates = int(data.get("snapshot_version", 0)), int(data.get("learner_updates", 0))
                for raw in data.get("creatures", []):
                    brain = LiveCreatureBrain(m1_registry=self.m1_registry); brain.m1.load_dict(raw["m1"])
                    record = CreatureRecord(str(raw["creature_id"]), BodyGenome.from_dict(raw["body"]), brain, str(raw["species_id"]), str(raw["m1_species_id"]), float(raw["born_seconds"]))
                    record.behavior.extend(float(x) for x in raw.get("behavior", [])); record.energy_totals.update({k: float(v) for k, v in raw.get("energy_totals", {}).items()}); record.death_penalty = float(raw.get("death_penalty", 0)); self.creatures[record.creature_id] = record
                for sample in data.get("replay", []): self.replay.append((_clone_batch(sample["batch"]), sample["target"].detach().cpu().clone()))
                self.body_champions = {key: Champion.from_dict(value) for key, value in data.get("champions", {}).items()}
                self._refresh_representatives()
                if len(self.creatures) == POPULATION: self.checkpoint_status = "Checkpoint loaded"; return
                raise ValueError("checkpoint population is incomplete")
            except Exception as exc:
                legacy = os.path.join(os.path.dirname(self.checkpoint_path), "legacy"); os.makedirs(legacy, exist_ok=True)
                try: shutil.move(self.checkpoint_path, os.path.join(legacy, os.path.basename(self.checkpoint_path) + ".v4"))
                except OSError: pass
                print(f"[ECOSYSTEM] checkpoint ignored: {exc}", flush=True)
                self.creatures.clear(); self.replay.clear()
        self._seed_population(); self.checkpoint_status = "Fresh v5 population"

    def _checkpoint_state(self) -> Dict[str, Any]:
        with self.lock, self.model_lock:
            return {"schema": SCHEMA_VERSION, "protocol_version": PROTOCOL_VERSION, "registry": deepcopy(self.registry.to_dict()), "m1_registry": deepcopy(self.m1_registry.to_dict()), "clock": self.clock, "next_cohort": self.next_cohort, "body_threshold": self.body_threshold, "m1_threshold": self.m1_threshold, "snapshot_version": self.snapshot_version, "learner_updates": self.learner_updates, "policy": self.policy.state_dict(), "m3": self.m3.state_dict(), "m3_optimizer": self.m3_optimizer.state_dict(), "creatures": [{"creature_id": r.creature_id, "body": r.body.to_dict(), "m1": r.brain.m1.to_dict(), "species_id": r.species_id, "m1_species_id": r.m1_species_id, "born_seconds": r.born_seconds, "behavior": list(r.behavior), "energy_totals": dict(r.energy_totals), "death_penalty": r.death_penalty} for r in self.creatures.values()], "replay": [{"batch": _clone_batch(batch), "target": target.detach().cpu().clone()} for batch, target in self.replay], "champions": {key: value.to_dict() for key, value in self.body_champions.items()}}
    def save(self) -> None:
        try:
            self.checkpoint_status = "Saving"; temporary = self.checkpoint_path + ".tmp"; os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True); torch.save(self._checkpoint_state(), temporary); os.replace(temporary, self.checkpoint_path); self.last_checkpoint = self.clock; self.checkpoint_status = "Saved"
        except Exception as exc: self.checkpoint_status = f"Save failed: {exc}"

    # Population and continuous body/M1 evolution.
    def _seed_population(self) -> None:
        for index, (body, copies) in enumerate(zip(seed_templates(self.registry), (2, 1, 1)), 1):
            for _ in range(copies): self._insert_new(body.clone(), species=f"body_seed_{index}", m1_species=f"body_seed_{index}_m1")
        self._refresh_representatives()
    def _refresh_representatives(self) -> None:
        self.body_species_representatives = {}
        for record in self.creatures.values(): self.body_species_representatives.setdefault(record.species_id, record.body.clone())
    def _new_species_id(self) -> str: return f"body_{uuid.uuid4().hex[:8]}"
    def _new_m1_species_id(self) -> str: return f"m1_{uuid.uuid4().hex[:8]}"
    def _assign_body_species(self, body: BodyGenome) -> str:
        for species, representative in self.body_species_representatives.items():
            if body_distance(body, representative) <= self.body_threshold: return species
        species = self._new_species_id(); self.body_species_representatives[species] = body.clone(); return species
    def _assign_m1_species(self, brain: NEATGoalSetter, body_species: str) -> str:
        for record in self.creatures.values():
            if record.species_id == body_species and m1_distance(brain, record.brain.m1) <= self.m1_threshold: return record.m1_species_id
        return self._new_m1_species_id()
    def _insert_new(self, body: BodyGenome, species: Optional[str] = None, m1_species: Optional[str] = None) -> CreatureRecord:
        body.validate(); brain = LiveCreatureBrain(m1_registry=self.m1_registry); sid = species or self._assign_body_species(body); record = CreatureRecord(str(uuid.uuid4()), body, brain, sid, m1_species or self._assign_m1_species(brain.m1, sid), self.clock); self.creatures[record.creature_id] = record; return record
    def _eligible_parent(self, record: CreatureRecord) -> bool: return record.age(self.clock) >= COHORT_SECONDS and len(record.behavior) >= MIN_PARENT_SAMPLES
    def _offspring(self, preferred_species: Optional[str] = None) -> CreatureRecord:
        pool = [r for r in self.creatures.values() if (not preferred_species or r.species_id == preferred_species)] or list(self.creatures.values()); parent = min(pool, key=lambda r: r.body_fitness())
        mates = [r for r in pool if r.creature_id != parent.creature_id]; mate = random.choice(mates) if mates else parent
        body = mutate_body(crossover_body(parent.body, parent.body_fitness(), mate.body, mate.body_fitness(), self.registry), self.registry); child = self._insert_new(body, self._assign_body_species(body))
        child.brain.m1 = NEATGoalSetter.crossover(parent.brain.m1, parent.energy_rate(self.clock), mate.brain.m1, mate.energy_rate(self.clock), self.m1_registry); child.brain.m1.mutate(weight_prob=.8, add_conn_prob=.1, add_node_prob=.05, activation_prob=.03); child.m1_species_id = self._assign_m1_species(child.brain.m1, child.species_id); return child
    def _replace_periodic(self) -> Tuple[List[CreatureRecord], List[CreatureRecord]]:
        candidates = [r for r in self.creatures.values() if self._eligible_parent(r)] or list(self.creatures.values()); victims = sorted(candidates, key=lambda r: (r.body_fitness(), -r.energy_rate(self.clock)), reverse=True)[:REPLACEMENTS]; children = []
        for victim in victims: self.creatures.pop(victim.creature_id, None); children.append(self._offspring(victim.species_id))
        self._refresh_representatives(); return victims, children

    # Protocol and observations.
    def _queue_command(self, kind: str, **payload: Any) -> None:
        command_id = uuid.uuid4().hex; self.pending_commands[command_id] = {"command_id": command_id, "type": kind, **payload}
    def _handle_events(self, events: Iterable[Dict[str, Any]]) -> List[str]:
        acknowledged = []
        for event in events:
            event_id = str(event.get("event_id", ""));
            if not event_id or event_id in self._processed_events: continue
            acknowledged.append(event_id); self._processed_events.add(event_id); self.processed_event_ids.append(event_id)
            if len(self.processed_event_ids) == self.processed_event_ids.maxlen: self._processed_events.discard(self.processed_event_ids[0])
            command_id = str(event.get("command_id", "")); self.pending_commands.pop(command_id, None)
            if event.get("type") == "death":
                dead = self.creatures.pop(str(event.get("creature_id", "")), None)
                if dead is not None: dead.death_penalty += 100. if event.get("reason") == "predator" else 0.; child = self._offspring(dead.species_id); self._queue_command("spawn", creature=self.spawn_payload(child))
        return acknowledged
    def initial_response(self) -> Dict[str, Any]:
        with self.lock:
            queued = {command.get("creature", {}).get("creature_id") for command in self.pending_commands.values() if command.get("type") == "spawn"}
            for record in self.creatures.values():
                if record.creature_id not in queued: self._queue_command("spawn", creature=self.spawn_payload(record))
            return self._response(-1, "", "", {}, [], [])
    def _response(self, tick: int, observation_id: str, action_id: str, actions: Dict[str, Any], event_ids: List[str], segment_ids: List[str]) -> Dict[str, Any]:
        return {"protocol_version": PROTOCOL_VERSION, "session_id": self.session_id, "source_tick": tick, "source_observation_id": observation_id, "action_id": action_id, "model_version": self.snapshot_version, "actions": actions, "commands": list(self.pending_commands.values()), "acknowledged_event_ids": event_ids, "acknowledged_segment_ids": segment_ids, "server_status": "Connected", "checkpoint_status": self.checkpoint_status, "telemetry": {"species": len({r.species_id for r in self.creatures.values()}), "learner_lag": self._queue_size_hint(), "learner_updates": self.learner_updates, "dropped_learning": 0, "m3_error": self.last_m3_error, "replay_size": len(self.replay), "motion_replay_size": sum(float(torch.linalg.vector_norm(target)) >= M3_MOTION_THRESHOLD for _, target in self.replay), "learner_busy": self.learner_busy, "observe_ms": self.last_observe_ms, "inference_ms": self.last_inference_ms, "learner_update_ms": self.last_learner_update_ms, "observe_tick": self.last_observe_tick, "inference_tick": self.last_inference_tick, "learner_tick": self.last_learner_tick}}
    def _queue_size_hint(self) -> int:
        try: return self.transition_queue.qsize()
        except NotImplementedError: return 0
    @staticmethod
    def _m1_inputs(raw: Dict[str, Any], goal: Tuple[float, float]) -> List[float]:
        get = lambda key: float(raw.get(key, 0.)); values = [get("energy") / 200, get("position_x") / 50, get("position_y") / 50, get("velocity_x") / 20, get("velocity_y") / 20, get("angular_velocity") / 360, get("rotation_sin"), get("rotation_cos"), get("sees_food"), get("rel_food_x") / 20, get("rel_food_y") / 20, get("sees_predator"), get("rel_predator_x") / 20, get("rel_predator_y") / 20]
        for neighbour in list(raw.get("neighbours", []))[:3]: values += [float(neighbour.get("x", 0)) / 20, float(neighbour.get("y", 0)) / 20, float(neighbour.get("same", 0)), float(neighbour.get("valid", 1))]
        while len(values) < 26: values += [0, 0, 0, 0]
        return (values[:26] + [goal[0] / .25, goal[1] / .25])[:28]
    @staticmethod
    def _update_energy(record: CreatureRecord, raw: Dict[str, Any]) -> None:
        for key in ("food_gain", "baseline_cost", "movement_cost", "idle_cost"):
            current = float(raw.get(key, record.last_energy_counters.get(key, 0.))); record.energy_totals[key] += max(0., current - record.last_energy_counters.get(key, current)); record.last_energy_counters[key] = current
    @staticmethod
    def _reward(displacement: torch.Tensor, goal: torch.Tensor, root: Dict[str, Any]) -> float:
        distance = float(torch.linalg.vector_norm(displacement)); goal_size = max(MIN_GOAL, float(torch.linalg.vector_norm(goal))); progress = float(torch.dot(displacement, goal) / goal_size); food = max(0., float(root.get("food_gain", 0.))); cost = float(root.get("movement_cost", 0.)) + float(root.get("idle_cost", 0.)); return max(-5., min(5., progress * 20. + distance * 3. + food * .05 - cost * .05 - (.25 if distance < M3_MOTION_THRESHOLD else 0.)))
    def _batch(self, record: CreatureRecord) -> Tuple[Dict[str, torch.Tensor], List[int]]:
        frames = list(record.history); frames = [frames[0]] * max(0, HISTORY - len(frames)) + frames
        limb_ids = sorted({int(limb["innovation_id"]) for frame in frames for limb in frame["limbs"]}) or [0]; count = len(limb_ids)
        states = torch.zeros(1, HISTORY, count, STATE_DIM); paths = torch.full((1, HISTORY, count, MAX_PATH_DEPTH), -1, dtype=torch.long); innovations = torch.zeros(1, HISTORY, count, dtype=torch.long); joint_types = torch.zeros(1, HISTORY, count, dtype=torch.long); roots = torch.zeros(1, HISTORY, ROOT_DIM); goals = torch.zeros(1, HISTORY, 2); mask = torch.zeros(1, HISTORY, count, dtype=torch.bool)
        for t, frame in enumerate(frames[-HISTORY:]):
            root, goal = frame["root"], frame["goal"]; roots[0, t] = torch.tensor([float(root.get("velocity_x", 0)) / 20, float(root.get("velocity_y", 0)) / 20, float(root.get("angular_velocity", 0)) / 360, float(root.get("rotation_sin", 0)), float(root.get("rotation_cos", 1)), float(root.get("energy", 0)) / 200, math.hypot(*goal) / .25, float(root.get("baseline_cost", 0)) / 100, float(root.get("movement_cost", 0)) / 100, float(root.get("idle_cost", 0)) / 100]); goals[0, t] = torch.tensor(goal); current = {int(limb["innovation_id"]): limb for limb in frame["limbs"]}
            for i, innovation in enumerate(limb_ids):
                limb = current.get(innovation)
                if limb is None: continue
                mask[0, t, i] = True; innovations[0, t, i] = innovation; joint_types[0, t, i] = int(limb.get("joint_type_id", 0)); states[0, t, i] = torch.tensor([float(limb.get("joint_angle", 0)) / 180, float(limb.get("angular_velocity", 0)) / 360, float(limb.get("touch_self", 0)), float(limb.get("touch_other_creature", 0)), float(limb.get("touch_environment", 0)), float(limb.get("previous_action", 0)) / MAX_DELTA_DEGREES, float(limb.get("width", 1)) / 3, float(limb.get("height", .5)) / 1.2, float(limb.get("max_torque", 50)) / 200, float(limb.get("depth", 1)) / 5, float(limb.get("min_angle", -45)) / 180, float(limb.get("max_angle", 45)) / 180, float(limb.get("at_limit", 0))]); path = [int(x) for x in limb.get("gene_path", [])[:MAX_PATH_DEPTH]]; paths[0, t, i, :len(path)] = torch.tensor(path) if path else paths[0, t, i, :0]
        return {"states": states, "paths": paths, "innovations": innovations, "joint_types": joint_types, "roots": roots, "goals": goals, "mask": mask}, limb_ids

    def observe(self, message: Dict[str, Any]) -> Dict[str, Any]:
        if int(message.get("protocol_version", 0)) != PROTOCOL_VERSION: raise ValueError(f"protocol_version must be {PROTOCOL_VERSION}")
        started = time.perf_counter(); tick = int(message.get("physics_tick", message.get("tick_id", 0))); observation_id = str(message.get("observation_id", "")); action_id = uuid.uuid4().hex
        with self.lock:
            self.clock += max(0., min(float(message.get("fixed_delta_time", .02)) * int(message.get("control_ticks", 4)), .5)); acknowledged = self._handle_events(message.get("lifecycle_events", [])); gap_actions = {str(item.get("action_id", "")) for item in message.get("executed_segments", []) if item.get("transport_gap")}; segment_ids = [str(item.get("segment_id", "")) for item in message.get("executed_segments", []) if item.get("segment_id")]
            raw_creatures = message.get("creatures", {}); batches, limbs_by_creature, roots = {}, {}, {}
            for creature_id, raw in raw_creatures.items():
                record = self.creatures.get(str(creature_id));
                if record is None: continue
                root, limbs = raw.get("global_inputs", {}), list(raw.get("limbs", []));
                if not limbs: continue
                current_root = (float(root.get("position_x", 0)), float(root.get("position_y", 0))); previous_id = str(message.get("previous_action_id", "")); pending = record.pending_actions.get(previous_id)
                if pending is not None and record.last_root is not None and previous_id not in gap_actions:
                    displacement = torch.tensor([current_root[0] - record.last_root[0], current_root[1] - record.last_root[1]]); reward = self._reward(displacement, pending.goal[0], root); record.behavior.append(reward); self.transition_queue.put(Transition(record.creature_id, _clone_batch(pending.batch), pending.action.clone(), pending.log_prob, pending.value, displacement.unsqueeze(0), reward, pending.source_tick))
                record.last_root = current_root; self._update_energy(record, root); executed_goal = tuple(float(x) for x in raw.get("current_goal", record.latest_goal)); record.history.append({"root": deepcopy(root), "limbs": deepcopy(limbs), "goal": executed_goal}); goal = record.brain.goal(self._m1_inputs(root, record.latest_goal)); record.latest_goal = (float(goal[0]), float(goal[1])); batch, limb_ids = self._batch(record); batch["goals"][:, -1] = torch.tensor(record.latest_goal); batches[record.creature_id], limbs_by_creature[record.creature_id], roots[record.creature_id] = batch, limb_ids, current_root; record.control_steps += 1
        inference_started = time.perf_counter(); inferred = {}
        for creature_id, batch in batches.items():
            seed = int.from_bytes(hashlib.sha256(f"{self.session_id}:{tick}:{creature_id}".encode()).digest()[:4], "big")
            inferred[creature_id] = (*self.inference_policy.act(batch, seed=seed, deterministic=False), seed)
        self.last_inference_ms = (time.perf_counter() - inference_started) * 1000
        with self.lock:
            actions = {}
            for creature_id, values in inferred.items():
                record = self.creatures.get(creature_id)
                if record is None: continue
                action, log_prob, value, entropy, seed = values; limb_ids = limbs_by_creature[creature_id]; deltas = {str(innovation): float(delta) for innovation, delta in zip(limb_ids, action.tolist()) if innovation != 0}; record.pending_actions[action_id] = PendingAction(_clone_batch(batches[creature_id]), action, log_prob, value, torch.tensor([record.latest_goal]), tick); record.pending_actions = {key: value for key, value in record.pending_actions.items() if tick - value.source_tick <= 64}
                actions[creature_id] = {"deltas": deltas, "goal": list(record.latest_goal), "sample_seed": seed, "log_probability": log_prob, "telemetry": {"species": record.species_id, "m1_species": record.m1_species_id, "age": record.age(self.clock), "body_fitness": None if not math.isfinite(record.body_fitness()) else record.body_fitness(), "energy_rate": record.energy_rate(self.clock), "learner_updates": self.learner_updates, "control_mode": "stochastic_M2", "policy_entropy": entropy}}
            while self.clock >= self.next_cohort:
                self.next_cohort += COHORT_SECONDS; victims, children = self._replace_periodic()
                for victim in victims: self._queue_command("despawn", creature_id=victim.creature_id)
                for child in children: self._queue_command("spawn", creature=self.spawn_payload(child))
            if self.clock - self.last_checkpoint >= CHECKPOINT_SECONDS: self.save()
            self.last_observe_ms = (time.perf_counter() - started) * 1000; self.last_observe_tick = self.last_inference_tick = tick
            return self._response(tick, observation_id, action_id, actions, acknowledged, segment_ids)

    def _publish_snapshot(self) -> None:
        with self.model_lock: self.inference_policy = self.policy.clone_for_inference(); self.snapshot_version += 1
    def _learner_loop(self) -> None:
        while not self.stop_event.is_set():
            try: first = self.transition_queue.get(timeout=.1)
            except queue.Empty: continue
            samples = [first]
            while len(samples) < M3_BATCH_SIZE:
                try: samples.append(self.transition_queue.get_nowait())
                except queue.Empty: break
            self._learn(samples)
    def _learn(self, transitions: List[Transition]) -> None:
        started = time.perf_counter(); self.learner_busy = True
        try:
            with self.model_lock:
                for item in transitions: self.replay.append((_clone_batch(item.batch), item.displacement.detach().cpu().clone()))
                samples = random.sample(list(self.replay), min(M3_BATCH_SIZE, len(self.replay)))
                if samples:
                    batch = pad_batches([item[0] for item in samples], device=DEVICE); targets = torch.cat([item[1] for item in samples]).to(DEVICE); self.m3.train(); self.m3_optimizer.zero_grad(set_to_none=True); prediction = self.m3(*(batch[key] for key in MODEL_KEYS), "m3"); loss = (prediction - targets).square().mean(); loss.backward(); self.m3_optimizer.step(); self.last_m3_error = float(loss.detach())
                policy_samples = [(item.batch, item.action, item.log_prob, item.reward, item.value) for item in transitions]
                self.policy.update(policy_samples); self.learner_updates += len(transitions); self.last_learner_tick = max(item.source_tick for item in transitions); self._publish_snapshot()
        finally:
            self.last_learner_update_ms = (time.perf_counter() - started) * 1000; self.learner_busy = False
    def spawn_payload(self, record: CreatureRecord) -> Dict[str, Any]: return {"creature_id": record.creature_id, "species_id": record.species_id, "m1_genome_id": f"{record.creature_id}:m1", "body_genome_id": record.body.genome_id, "body": record.body.to_dict()}
    def shutdown(self) -> None: self.stop_event.set(); self.learner.join(timeout=2); self.save()
