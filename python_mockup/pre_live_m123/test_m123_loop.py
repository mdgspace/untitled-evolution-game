"""
Automated tests for the M1→M2→M3 pipeline (Goal 1: fixed morphology).

Run with:
    python test_m123_loop.py
"""

import torch
import torch.nn.functional as F
from achiever_transformer import AchieverTransformer
from creature_brain_m123 import CreatureBrainM123


def test_achiever_transformer_basic():
    print("--- Test 1: AchieverTransformer Basic Forward Pass ---")
    model = AchieverTransformer(
        catalog_size=32, val_dim=4, d_model=32, nhead=2, num_layers=1, out_dim=3
    )

    # 3 joint tokens with 4-D values (angle, touch_s, touch_o, touch_e)
    cat_ids = torch.tensor([[1, 5, 12]])     # [1, 3]  long
    values  = torch.randn(1, 3, 4)           # [1, 3, 4]  float

    per_token_out = model(cat_ids, values, mode="per_token")
    global_out    = model(cat_ids, values, mode="global")

    print(f"Per-token output shape : {per_token_out.shape}  (expected [1, 3, 1])")
    print(f"Global output shape    : {global_out.shape}  (expected [1, 3])")
    assert per_token_out.shape == (1, 3, 1), f"Shape mismatch: {per_token_out.shape}"
    assert global_out.shape    == (1, 3),    f"Shape mismatch: {global_out.shape}"
    print("PASS: AchieverTransformer basic forward pass.\n")


def test_m123_inner_loop_and_m3_learning():
    print("--- Test 2: M123 Inner-Loop Action Optimisation & M3 Training ---")
    brain = CreatureBrainM123(
        catalog_size=32,
        d_model=32,
        inner_loop_steps=5,
        inner_lr=0.05,
        m3_lr=0.01,
        goal_setter_input_dim=30,
        nhead=2,
        num_layers=1,
        dim_feedforward=64,
    )

    # Morphology: 4 joints (catalog IDs 2..5)
    joint_ids = [2, 3, 4, 5]

    # 4-D joint sensor data: [angle, touch_self, touch_other, touch_env]
    # This matches the production API in server.py / creature_brain_m123.py
    joint_sensor_data = [
        [ 0.1, 0.0, 0.0, 1.0],
        [-0.2, 1.0, 0.0, 0.0],
        [ 0.5, 0.0, 0.0, 0.0],
        [ 0.0, 0.0, 0.0, 1.0],
    ]

    goal = torch.tensor([1.0, 0.5, 0.0])

    print("Running M2 action selection (inner-loop optimisation)...")
    actions, telemetry_m2 = brain.select_action(joint_ids, joint_sensor_data, goal, inner_steps=10)
    print(f"  Joint action deltas : {[round(a, 4) for a in actions.tolist()]}")
    print(f"  M3->goal loss       : {telemetry_m2['m3_goal_loss']:.6f}")
    assert len(actions) == 4, f"Expected 4 actions, got {len(actions)}"

    # Synthetic physics: real Δpos = sum(actions) * [0.1, 0.1, 0.0]
    s = actions.sum().item()
    real_delta_pos = torch.tensor([s * 0.1, s * 0.1, 0.0])

    print("\nTraining M3 dynamics predictor with real transition (21 steps)...")
    initial_loss, _ = brain.update_m3_dynamics(joint_ids, actions, real_delta_pos)
    loss = initial_loss
    for _ in range(20):
        loss, _ = brain.update_m3_dynamics(joint_ids, actions, real_delta_pos)

    print(f"  M3 dynamics loss: initial={initial_loss:.6f}  ->  final={loss:.6f}")
    assert loss < initial_loss, \
        f"M3 should converge: {initial_loss:.6f} → {loss:.6f}"
    print("PASS: M3 dynamics training converges.\n")

    print("Re-running action selection after M3 has been trained...")
    actions_post, telem_post = brain.select_action(joint_ids, joint_sensor_data, goal, inner_steps=10)
    print(f"  Post-training deltas: {[round(a, 4) for a in actions_post.tolist()]}")
    print(f"  Post M3->goal loss  : {telem_post['m3_goal_loss']:.6f}")
    print("PASS: M123 inner-loop & M3 learning.\n")


def test_neat_goal_setter():
    print("--- Test 3: NEAT Goal-Setter forward + mutation ---")
    from neat_goal_setter import NEATGoalSetter

    gs = NEATGoalSetter(input_dim=30, output_dim=3)
    inp = [0.1] * 30
    goal = gs.forward(inp)
    assert goal.shape == (3,), f"Goal shape: {goal.shape}"
    print(f"  Initial goal: {goal.tolist()}")

    # Check mutation doesn't crash and still produces valid output
    for _ in range(50):
        gs.mutate(weight_prob=0.8, add_node_prob=0.1, add_conn_prob=0.2)
    goal_mutated = gs.forward(inp)
    assert goal_mutated.shape == (3,)
    print(f"  Post-mutation goal: {goal_mutated.tolist()}")
    print("PASS: NEAT goal setter forward + mutation.\n")


def test_cls_token_m3_grad_flow():
    """Verify that the M2 inner loop gradient does flow back through M3 into M2."""
    print("--- Test 4: Gradient flow M3 CLS → M2 ---")
    brain = CreatureBrainM123(
        catalog_size=16, d_model=16, inner_loop_steps=1,
        inner_lr=0.01, m3_lr=0.001,
        goal_setter_input_dim=30, nhead=2, num_layers=1, dim_feedforward=32,
    )
    joint_ids = [1, 2]
    sensor_data = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    goal = torch.tensor([1.0, 0.0, 0.0])

    # Snapshot M2 params before
    before = {n: p.clone() for n, p in brain.m2.named_parameters()}

    brain.select_action(joint_ids, sensor_data, goal, inner_steps=3)

    # At least one M2 parameter should have changed
    changed = any(
        not torch.allclose(before[n], p)
        for n, p in brain.m2.named_parameters()
    )
    assert changed, "M2 parameters did not change — gradient flow broken!"
    print("PASS: M2 parameters updated via M3 gradient.\n")


if __name__ == "__main__":
    test_achiever_transformer_basic()
    test_m123_inner_loop_and_m3_learning()
    test_neat_goal_setter()
    test_cls_token_m3_grad_flow()
    print("=" * 50)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 50)
