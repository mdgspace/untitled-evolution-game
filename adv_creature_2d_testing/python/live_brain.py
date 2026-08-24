"""Four-frame morphology-aware M2/M3 models for the real-time ecosystem."""
from __future__ import annotations

from copy import deepcopy
import os
import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from neat_goal_setter import NEATGoalSetter, NEATInnovationRegistry

HISTORY = 4
MAX_PATH_DEPTH = 5
SLOTS = 8
MAX_DELTA_DEGREES = 4.0
STATE_DIM = 13
ROOT_DIM = 10
INNOVATION_BUCKETS = 4096


def _select_device() -> torch.device:
    """Select the accelerator once, with an explicit CPU escape hatch."""
    requested = os.environ.get("LIVE_ECOSYSTEM_DEVICE", "auto").strip().lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("LIVE_ECOSYSTEM_DEVICE requests CUDA, but this Torch build has no CUDA support")
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEVICE = _select_device()


def module_device(module: nn.Module) -> torch.device:
    return next(module.parameters()).device


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device = DEVICE) -> Dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def optimizer_to(optimizer: optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device, non_blocking=True)


class TemporalAchiever(nn.Module):
    """Compact masked self-attention over four frames and current control tokens."""
    def __init__(self, d_model: int = 16, nhead: int = 2, num_layers: int = 1, output_dim: int = 1):
        super().__init__()
        self.state = nn.Linear(STATE_DIM, d_model)
        self.root = nn.Linear(ROOT_DIM, d_model)
        self.goal = nn.Linear(2, d_model)
        self.path = nn.ModuleList([nn.Embedding(SLOTS + 1, d_model) for _ in range(MAX_PATH_DEPTH)])
        self.innovation = nn.Embedding(INNOVATION_BUCKETS, d_model)
        self.joint_type = nn.Embedding(4, d_model)
        self.time = nn.Embedding(HISTORY, d_model)
        self.token_type = nn.Embedding(4, d_model)  # limb, root, goal, cls
        layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 2, dropout=0.0,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.action_head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)
        self.global_head = nn.Linear(d_model, output_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))

    def _path_encoding(self, paths: torch.Tensor) -> torch.Tensor:
        encoded = 0
        for depth, table in enumerate(self.path):
            encoded = encoded + table(paths[..., depth].clamp(min=-1, max=SLOTS - 1) + 1)
        return encoded

    def encode(self, states: torch.Tensor, paths: torch.Tensor, innovations: torch.Tensor,
               joint_types: torch.Tensor, roots: torch.Tensor, goals: torch.Tensor,
               limb_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, frames, limbs, _ = states.shape
        time = self.time(torch.arange(frames, device=states.device)).view(1, frames, 1, -1)
        limb = (self.state(states) + self._path_encoding(paths)
                + self.innovation(innovations.remainder(INNOVATION_BUCKETS))
                + self.joint_type(joint_types.clamp(0, 3)) + time + self.token_type.weight[0])
        root = self.root(roots).unsqueeze(2) + time + self.token_type.weight[1]
        frame_tokens = torch.cat((root, limb), dim=2).reshape(batch, frames * (limbs + 1), -1)
        root_mask = torch.zeros(batch, frames, 1, dtype=torch.bool, device=states.device)
        frame_padding = torch.cat((root_mask, ~limb_mask), dim=2).reshape(batch, frames * (limbs + 1))
        goal = self.goal(goals[:, -1]).unsqueeze(1) + self.time.weight[frames - 1] + self.token_type.weight[2]
        cls = self.cls.expand(batch, -1, -1) + self.token_type.weight[3]
        tokens = torch.cat((cls, frame_tokens, goal), dim=1)
        padding = torch.cat((torch.zeros(batch, 1, dtype=torch.bool, device=states.device), frame_padding,
                             torch.zeros(batch, 1, dtype=torch.bool, device=states.device)), dim=1)
        return tokens, padding

    def forward(self, states: torch.Tensor, paths: torch.Tensor, innovations: torch.Tensor,
                joint_types: torch.Tensor, roots: torch.Tensor, goals: torch.Tensor,
                limb_mask: torch.Tensor, mode: str) -> torch.Tensor:
        # Keep tests and CPU-originating sensor batches usable after the model
        # is moved to CUDA. Runtime callers pre-stage batches on DEVICE so this
        # is a no-op on the hot path.
        device = module_device(self)
        if states.device != device:
            states, paths, innovations, joint_types = (value.to(device, non_blocking=True)
                                                       for value in (states, paths, innovations, joint_types))
            roots, goals, limb_mask = (value.to(device, non_blocking=True)
                                       for value in (roots, goals, limb_mask))
        tokens, padding = self.encode(states, paths, innovations, joint_types, roots, goals, limb_mask)
        encoded = self.encoder(tokens, src_key_padding_mask=padding)
        if mode == "m2":
            batch, frames, limbs, _ = states.shape
            start = 1 + (frames - 1) * (limbs + 1) + 1
            return torch.tanh(self.action_head(encoded[:, start:start + limbs]).squeeze(-1)) * MAX_DELTA_DEGREES
        if mode == "m3":
            return self.global_head(encoded[:, 0])
        raise ValueError(f"unknown mode {mode}")


MODEL_KEYS = ("states", "paths", "innovations", "joint_types", "roots", "goals", "mask")


class LiveCreatureBrain:
    def __init__(self, m1_input_dim: int = 28, d_model: int = 16,
                 m1_registry: NEATInnovationRegistry | None = None):
        self.m1 = NEATGoalSetter(m1_input_dim, 2, m1_registry)
        self.m2 = TemporalAchiever(d_model=d_model).to(DEVICE)
        self.optimizer = optim.Adam(self.m2.parameters(), lr=.002)
        self.updates = 0

    def goal(self, values: Sequence[float]) -> torch.Tensor:
        return self.m1.forward(list(values)) * .25

    def train_m2(self, batch: Dict[str, torch.Tensor], m3: TemporalAchiever) -> Tuple[float, torch.Tensor]:
        batch = move_batch(batch, module_device(self.m2))
        self.m2.train()
        self.optimizer.zero_grad(set_to_none=True)
        m3.zero_grad(set_to_none=True)
        original = [parameter.requires_grad for parameter in m3.parameters()]
        try:
            for parameter in m3.parameters():
                parameter.requires_grad_(False)
            actions = self.m2(*(batch[key] for key in MODEL_KEYS), "m2")
            m3_states = batch["states"].clone()
            m3_states[:, -1, :, 5] = actions / MAX_DELTA_DEGREES
            predicted = m3(m3_states, batch["paths"], batch["innovations"], batch["joint_types"],
                           batch["roots"], batch["goals"], batch["mask"], "m3")
            target = batch["goals"][:, -1]
            denominator = torch.clamp(target.square().mean(dim=-1), min=1e-4)
            losses = (predicted - target).square().mean(dim=-1) / denominator
            loss = losses.mean()
            loss.backward()
            self.optimizer.step()
        finally:
            for parameter, enabled in zip(m3.parameters(), original):
                parameter.requires_grad_(enabled)
            m3.zero_grad(set_to_none=True)
        self.updates += 1
        return float(loss.detach()), actions.detach()[0]

    def state_dict(self) -> Dict:
        return {"m1": self.m1.to_dict(), "m2": self.m2.state_dict(),
                "optimizer": self.optimizer.state_dict(), "updates": self.updates}

    def load_state_dict(self, value: Dict):
        self.m1.load_dict(value["m1"])
        self.m2.load_state_dict(value["m2"])
        self.optimizer.load_state_dict(value["optimizer"])
        optimizer_to(self.optimizer, module_device(self.m2))
        self.updates = int(value.get("updates", 0))


class BatchedInferenceSnapshot:
    """Immutable M2 models for the small live population.

    PyTorch's current Windows build falls back to a very slow generic path
    when ``vmap`` wraps TransformerEncoder.  Four regular encoder forwards are
    both predictable and substantially less disruptive to the Unity process.
    """
    def __init__(self, brains: Dict[str, LiveCreatureBrain], version: int):
        self.version = version
        self.ids = list(brains)
        self.models = {cid: deepcopy(brains[cid].m2).eval() for cid in self.ids}

    def act(self, batches: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        active = [cid for cid in self.ids if cid in batches]
        if not active:
            return {}
        with torch.no_grad():
            return {cid: self.models[cid](*(batches[cid][key] for key in MODEL_KEYS), "m2")[0]
                    for cid in active}


def pad_batches(items: List[Dict[str, torch.Tensor]], device: torch.device | None = None) -> Dict[str, torch.Tensor]:
    if not items:
        raise ValueError("cannot pad an empty batch")
    device = device or items[0]["states"].device
    maximum = max(item["states"].shape[2] for item in items)
    batch = len(items)
    result = {
        "states": torch.zeros(batch, HISTORY, maximum, STATE_DIM, device=device),
        "paths": torch.full((batch, HISTORY, maximum, MAX_PATH_DEPTH), -1, dtype=torch.long, device=device),
        "innovations": torch.zeros(batch, HISTORY, maximum, dtype=torch.long, device=device),
        "joint_types": torch.zeros(batch, HISTORY, maximum, dtype=torch.long, device=device),
        "roots": torch.zeros(batch, HISTORY, ROOT_DIM, device=device),
        "goals": torch.zeros(batch, HISTORY, 2, device=device),
        "mask": torch.zeros(batch, HISTORY, maximum, dtype=torch.bool, device=device),
    }
    for index, item in enumerate(items):
        count = item["states"].shape[2]
        for key in ("states", "paths", "innovations", "joint_types", "mask"):
            result[key][index, :, :count] = item[key][0].to(device, non_blocking=True)
        result["roots"][index] = item["roots"][0].to(device, non_blocking=True)
        result["goals"][index] = item["goals"][0].to(device, non_blocking=True)
    return result


class SharedStochasticPolicy:
    """Population-wide morphology-conditioned actor/critic.

    Exploration is sampled from this policy itself.  There is deliberately no
    scripted gait, calibration pattern, or action oscillator in this class.
    """
    def __init__(self, d_model: int = 16):
        self.model = TemporalAchiever(d_model=d_model, output_dim=2).to(DEVICE)
        self.value_head = nn.Linear(d_model, 1).to(DEVICE)
        self.log_std = nn.Parameter(torch.tensor(math.log(.8), dtype=torch.float32, device=DEVICE))
        self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.value_head.parameters()) + [self.log_std], lr=.001)

    def _forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        values = move_batch(batch, DEVICE)
        tokens, padding = self.model.encode(*(values[key] for key in MODEL_KEYS))
        encoded = self.model.encoder(tokens, src_key_padding_mask=padding)
        limbs = values["states"].shape[2]
        start = 1 + (HISTORY - 1) * (limbs + 1) + 1
        means = torch.tanh(self.model.action_head(encoded[:, start:start + limbs]).squeeze(-1)) * MAX_DELTA_DEGREES
        return means, self.value_head(encoded[:, 0]).squeeze(-1)

    def act(self, batch: Dict[str, torch.Tensor], seed: int, deterministic: bool = False) -> Tuple[torch.Tensor, float, float, float]:
        self.model.eval()
        with torch.no_grad():
            means, value = self._forward(batch)
            std = self.log_std.exp().clamp(.05, 2.0)
            if deterministic:
                actions = means
            else:
                generator = torch.Generator(device=DEVICE)
                generator.manual_seed(int(seed) & 0x7fffffff)
                actions = means + torch.randn(means.shape, generator=generator, device=DEVICE) * std
            actions = actions.clamp(-MAX_DELTA_DEGREES, MAX_DELTA_DEGREES)
            distribution = torch.distributions.Normal(means, std)
            log_prob = distribution.log_prob(actions).sum(dim=-1)
            entropy = distribution.entropy().sum(dim=-1)
            return actions[0].detach().cpu(), float(log_prob[0]), float(value[0]), float(entropy[0])

    def update(self, samples: List[Tuple[Dict[str, torch.Tensor], torch.Tensor, float, float, float]]) -> float:
        """One measured-return actor/critic update.

        Items contain (state, executed_action, behavior_log_prob, return,
        value_at_action).  The clipped ratio prevents stale snapshots from
        dominating a continuously published policy.
        """
        if not samples:
            return 0.0
        self.model.train()
        losses = []
        for batch, action, old_log_prob, reward, old_value in samples:
            means, value = self._forward(batch)
            std = self.log_std.exp().clamp(.05, 2.0)
            distribution = torch.distributions.Normal(means, std)
            current_log_prob = distribution.log_prob(action.to(DEVICE)).sum(dim=-1)
            ratio = torch.exp((current_log_prob - float(old_log_prob)).clamp(-10, 10))
            advantage = torch.tensor(float(reward) - float(old_value), device=DEVICE).clamp(-10, 10)
            unclipped = ratio * advantage
            clipped = torch.clamp(ratio, .8, 1.2) * advantage
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = F.smooth_l1_loss(value, torch.tensor([float(reward)], device=DEVICE))
            entropy = distribution.entropy().sum(dim=-1).mean()
            losses.append(policy_loss + .5 * value_loss - .01 * entropy)
        loss = torch.stack(losses).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(self.value_head.parameters()) + [self.log_std], 1.0)
        self.optimizer.step()
        return float(loss.detach())

    def state_dict(self) -> Dict:
        return {"model": self.model.state_dict(), "value": self.value_head.state_dict(), "log_std": self.log_std.detach().cpu(), "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, value: Dict) -> None:
        self.model.load_state_dict(value["model"])
        self.value_head.load_state_dict(value["value"])
        self.log_std.data.copy_(value["log_std"].to(DEVICE))
        self.optimizer.load_state_dict(value["optimizer"])
        optimizer_to(self.optimizer, DEVICE)

    def clone_for_inference(self) -> "SharedStochasticPolicy":
        clone = SharedStochasticPolicy()
        clone.model.load_state_dict(deepcopy(self.model.state_dict()))
        clone.value_head.load_state_dict(deepcopy(self.value_head.state_dict()))
        clone.log_std.data.copy_(self.log_std.detach())
        clone.model.eval()
        return clone


def train_population_m2(items: List[Tuple],
                        m3: TemporalAchiever) -> List[float]:
    """Train M2 through frozen M3 plus measured-world action reinforcement."""
    if not items:
        return []
    m3.zero_grad(set_to_none=True)
    original = [parameter.requires_grad for parameter in m3.parameters()]
    try:
        for parameter in m3.parameters():
            parameter.requires_grad_(False)
        losses: List[float] = []
        for item in items:
            brain, batch = item[0], item[1]
            batch = move_batch(batch, module_device(brain.m2))
            executed_batch = item[2] if len(item) >= 3 else None
            locomotion_reward = float(item[3]) if len(item) >= 4 else 0.0
            brain.optimizer.zero_grad(set_to_none=True)
            actions = brain.m2(*(batch[key] for key in MODEL_KEYS), "m2")
            m3_states = batch["states"].clone()
            m3_states[:, -1, :, 5] = actions / MAX_DELTA_DEGREES
            predicted = m3(m3_states, batch["paths"], batch["innovations"], batch["joint_types"],
                           batch["roots"], batch["goals"], batch["mask"], "m3")
            targets = batch["goals"][:, -1]
            denominator = torch.clamp(targets.square().mean(), min=1e-4)
            model_loss = (predicted - targets).square().mean() / denominator
            valid = batch["mask"][:, -1].float()
            previous_actions = batch["states"][:, -1, :, 5] * MAX_DELTA_DEGREES
            smoothness = ((actions - previous_actions).square() * valid).sum() / valid.sum().clamp_min(1.0)
            magnitude = ((actions / MAX_DELTA_DEGREES).square() * valid).sum() / valid.sum().clamp_min(1.0)
            imitation = torch.zeros((), dtype=actions.dtype, device=actions.device)
            if executed_batch is not None and locomotion_reward > 0.0:
                executed = executed_batch["states"][:, -1, :, 5].to(actions.device, non_blocking=True) * MAX_DELTA_DEGREES
                imitation = (F.smooth_l1_loss(actions, executed, reduction="none") * valid).sum() / valid.sum().clamp_min(1.0)
            # Only actions that produced measured, goal-aligned displacement
            # are imitated.  A small action-rate penalty suppresses flailing.
            loss = (model_loss + .35 * min(2.0, max(0.0, locomotion_reward)) * imitation
                    + .002 * smoothness + .05 * magnitude)
            loss.backward()
            brain.optimizer.step()
            brain.updates += 1
            losses.append(float(model_loss.detach()))
        return losses
    finally:
        for parameter, enabled in zip(m3.parameters(), original):
            parameter.requires_grad_(enabled)
        m3.zero_grad(set_to_none=True)
