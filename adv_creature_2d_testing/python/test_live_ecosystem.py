"""Production-contract tests for protocol v3, learning, genetics, and persistence."""
import os
import tempfile
import time

import torch

from genome import InnovationRegistry, body_distance, mutate_body, seed_templates
from live_brain import (DEVICE, HISTORY, MAX_DELTA_DEGREES, MODEL_KEYS, ROOT_DIM, STATE_DIM,
                        BatchedInferenceSnapshot, LiveCreatureBrain, SharedStochasticPolicy,
                        TemporalAchiever, train_population_m2)
from live_ecosystem import (MIN_PARENT_SAMPLES, PROTOCOL_VERSION, REPLACEMENTS,
                            LiveEcosystem)
from neat_goal_setter import NEATInnovationRegistry


def temporal_batch(limbs=3, goal=(.1, 0.0)):
    values = {
        "states": torch.zeros(1, HISTORY, limbs, STATE_DIM),
        "paths": torch.zeros(1, HISTORY, limbs, 5, dtype=torch.long),
        "innovations": torch.arange(1, limbs + 1).view(1, 1, limbs).expand(1, HISTORY, limbs).clone(),
        "joint_types": torch.zeros(1, HISTORY, limbs, dtype=torch.long),
        "roots": torch.zeros(1, HISTORY, ROOT_DIM),
        "goals": torch.zeros(1, HISTORY, 2),
        "mask": torch.ones(1, HISTORY, limbs, dtype=torch.bool),
    }
    values["goals"][:, -1] = torch.tensor(goal)
    return values


def test_structural_innovations_and_seed_species():
    registry = InnovationRegistry()
    templates = seed_templates(registry)
    clones = [template.clone() for template in templates]
    assert templates[0].paths() == clones[0].paths()
    assert {gene.innovation_id for gene in templates[0].limbs}.issubset(
        {gene.innovation_id for gene in templates[1].limbs})
    assert body_distance(templates[0], templates[0].clone()) == 0
    assert len(mutate_body(templates[0], registry).limbs) >= 1


def test_attention_shapes_bounds_and_m3_gradient_freeze():
    registry = NEATInnovationRegistry()
    brain = LiveCreatureBrain(m1_registry=registry, d_model=16)
    m3 = TemporalAchiever(d_model=16, nhead=4, num_layers=1, output_dim=2).to(DEVICE)
    batch = temporal_batch()
    with torch.no_grad():
        action = brain.m2(*(batch[key] for key in MODEL_KEYS), "m2")
    assert action.shape == (1, 3)
    assert torch.all(action <= MAX_DELTA_DEGREES) and torch.all(action >= -MAX_DELTA_DEGREES)
    before = {name: value.detach().clone() for name, value in m3.named_parameters()}
    brain.train_m2(batch, m3)
    assert all(torch.equal(before[name], value.detach()) for name, value in m3.named_parameters())
    assert all(parameter.grad is None for parameter in m3.parameters())


def test_shared_policy_explores_without_a_scripted_gait():
    policy = SharedStochasticPolicy(d_model=16)
    batch = temporal_batch()
    action_a, log_prob_a, value_a, entropy_a = policy.act(batch, seed=11)
    action_b, _, _, _ = policy.act(batch, seed=12)
    assert torch.all(action_a <= MAX_DELTA_DEGREES) and torch.all(action_a >= -MAX_DELTA_DEGREES)
    assert entropy_a > 0 and isinstance(log_prob_a, float) and isinstance(value_a, float)
    assert not torch.equal(action_a, action_b), "exploration must be sampled by M2, not a fixed pattern"


def test_vectorized_snapshot_is_immutable_and_handles_morphologies():
    brains = {"a": LiveCreatureBrain(d_model=16), "b": LiveCreatureBrain(d_model=16)}
    snapshot = BatchedInferenceSnapshot(brains, 7)
    output = snapshot.act({"a": temporal_batch(2), "b": temporal_batch(4)})
    assert output["a"].shape == (2,) and output["b"].shape == (4,)
    saved = output["a"].clone()
    with torch.no_grad():
        next(brains["a"].m2.parameters()).add_(100)
    assert torch.allclose(saved, snapshot.act({"a": temporal_batch(2)})["a"], atol=1e-5)
    m3 = TemporalAchiever(d_model=16, nhead=4, num_layers=1, output_dim=2).to(DEVICE)
    before = [[parameter.detach().clone() for parameter in brain.m2.parameters()]
              for brain in brains.values()]
    losses = train_population_m2([(brains["a"], temporal_batch(2)), (brains["b"], temporal_batch(4))], m3)
    assert len(losses) == 2
    assert all(any(not torch.equal(old_parameter, new_parameter.detach())
                   for old_parameter, new_parameter in zip(old_parameters, brain.m2.parameters()))
               for old_parameters, brain in zip(before, brains.values()))
    assert all(parameter.grad is None for parameter in m3.parameters())


def test_protocol_temporal_alignment_turnover_and_checkpoint():
    directory = tempfile.mkdtemp()
    checkpoint = os.path.join(directory, "live_ecosystem_v4.pt")
    ecosystem = LiveEcosystem(checkpoint)
    try:
        assert len(ecosystem.creatures) == 4
        assert len({record.species_id for record in ecosystem.creatures.values()}) == 3
        initial = ecosystem.initial_response()
        assert initial["protocol_version"] == PROTOCOL_VERSION
        assert len(initial["commands"]) == 4
        startup = initial["commands"][0]
        ack = ecosystem.observe({"protocol_version": PROTOCOL_VERSION, "tick_id": 0,
                                 "fixed_delta_time": .02, "control_ticks": 3, "creatures": {},
                                 "lifecycle_events": [{"event_id": "spawn-ack-1", "type": "spawn_result",
                                                       "command_id": startup["command_id"], "success": True}]})
        assert ack["acknowledged_event_ids"] == ["spawn-ack-1"]
        assert startup["command_id"] not in ecosystem.pending_commands

        creature = next(iter(ecosystem.creatures.values()))
        limb = creature.body.limbs[0]
        def observation(position, previous_action, tick):
            return {"protocol_version": PROTOCOL_VERSION, "tick_id": tick, "physics_tick": tick,
                    "observation_id": f"obs-{tick}", "previous_action_id": previous_action,
                    "fixed_delta_time": .02, "control_ticks": 4, "lifecycle_events": [], "executed_segments": [],
                    "creatures": {creature.creature_id: {"current_goal": [.1, 0],
                        "global_inputs": {"position_x": position, "position_y": 0, "energy": 100,
                                          "rotation_cos": 1, "food_gain": 0, "baseline_cost": position,
                                          "movement_cost": 0, "idle_cost": 0},
                        "limbs": [{"innovation_id": limb.innovation_id, "gene_path": list(creature.body.paths()[limb.innovation_id]),
                                   "joint_angle": 0, "angular_velocity": 0, "previous_action": 0,
                                   "width": limb.width, "height": limb.height, "max_torque": limb.max_torque,
                                   "depth": 1}]}}}
        first = ecosystem.observe(observation(0, "", 3))
        assert first["source_observation_id"] == "obs-3"
        assert first["action_id"] and first["actions"]
        assert all("sample_seed" in action and "log_probability" in action for action in first["actions"].values())
        action_id = first["action_id"]
        ecosystem.observe(observation(.2, action_id, 7))
        deadline = time.time() + 5
        while not ecosystem.replay and time.time() < deadline: time.sleep(.02)
        assert ecosystem.replay and torch.allclose(ecosystem.replay[-1][1], torch.tensor([[.2, 0.0]]))

        ecosystem.clock = 61
        for record in ecosystem.creatures.values():
            record.born_seconds = 0
            record.behavior.extend([1.0] * MIN_PARENT_SAMPLES)
        victims, children = ecosystem._replace_periodic()
        assert len(victims) == REPLACEMENTS and len(children) == REPLACEMENTS and len(ecosystem.creatures) == 4
        ecosystem.save()
    finally:
        ecosystem.shutdown()

    restored = LiveEcosystem(checkpoint)
    try:
        assert len(restored.creatures) == 4
        assert restored.replay
        assert restored.registry.structural
        assert restored.snapshot_version >= 0
    finally:
        restored.shutdown()


if __name__ == "__main__":
    test_structural_innovations_and_seed_species()
    test_attention_shapes_bounds_and_m3_gradient_freeze()
    test_shared_policy_explores_without_a_scripted_gait()
    test_vectorized_snapshot_is_immutable_and_handles_morphologies()
    test_protocol_temporal_alignment_turnover_and_checkpoint()
    print("live ecosystem tests passed")
