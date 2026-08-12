import torch
import torch.nn.functional as F
from achiever_transformer import AchieverTransformer
from creature_brain_m123 import CreatureBrainM123

def test_achiever_transformer_basic():
    print("--- Test 1: AchieverTransformer Basic Forward Pass ---")
    model = AchieverTransformer(catalog_size=32, val_dim=3, d_model=32, nhead=2, num_layers=1, out_dim=3)
    
    # 3 joint tokens (e.g. joint IDs 1, 5, 12)
    cat_ids = torch.tensor([[1, 5, 12]]) # [1, 3]
    values = torch.randn(1, 3, 3)         # [1, 3, 3]
    
    per_token_out = model(cat_ids, values, mode="per_token")
    global_out = model(cat_ids, values, mode="global")
    
    print(f"Per-token output shape: {per_token_out.shape} (Expected: [1, 3, 1])")
    print(f"Global output shape:    {global_out.shape} (Expected: [1, 3])")
    assert per_token_out.shape == (1, 3, 1)
    assert global_out.shape == (1, 3)
    print("PASS: AchieverTransformer basic forward pass.")

def test_m123_inner_loop_and_m3_learning():
    print("\n--- Test 2: M123 Inner-Loop Action Optimization & M3 Training ---")
    brain = CreatureBrainM123(catalog_size=32, d_model=32, inner_loop_steps=5, inner_lr=0.05, m3_lr=0.01)
    
    # Morphology with 4 joints (joint catalog IDs: [2, 3, 4, 5])
    joint_ids = [2, 3, 4, 5]
    joint_angles = [0.1, -0.2, 0.5, 0.0]
    
    # Target goal in 3D: [1.0, 0.5, 0.0]
    goal = torch.tensor([1.0, 0.5, 0.0])
    
    print("Running M2 action selection (inner-loop optimization)...")
    actions, telemetry_m2 = brain.select_action(joint_ids, joint_angles, goal, inner_steps=10)
    print(f"Computed joint action deltas: {actions.tolist()}")
    print(f"Telemetry M2 M3 Goal Loss: {telemetry_m2['m3_goal_loss']:.6f}")
    assert len(actions) == 4
    
    # Simulate synthetic physics transition: delta_pos = sum(actions) * [0.1, 0.1, 0.0]
    real_delta_pos = torch.tensor([actions.sum().item() * 0.1, actions.sum().item() * 0.1, 0.0])
    
    print("Updating M3 dynamics predictor with real transition...")
    initial_loss, initial_m3_telem = brain.update_m3_dynamics(joint_ids, actions, real_delta_pos)
    
    # Train M3 for 20 synthetic steps to verify M3 convergence
    for _ in range(20):
        loss, m3_telem = brain.update_m3_dynamics(joint_ids, actions, real_delta_pos)
        
    print(f"M3 dynamics loss: initial={initial_loss:.6f} -> final={loss:.6f}")
    assert loss < initial_loss
    print("PASS: M3 dynamics training converges.")
    
    # Re-run action selection now that M3 is trained
    print("Re-running action selection post M3 training...")
    actions_post, telemetry_post = brain.select_action(joint_ids, joint_angles, goal, inner_steps=10)
    print(f"Post-training joint action deltas: {actions_post.tolist()}")
    print("PASS: M123 Inner Loop & M3 Learning.")

if __name__ == "__main__":
    test_achiever_transformer_basic()
    test_m123_inner_loop_and_m3_learning()
    print("\nALL GOAL 1 TESTS PASSED SUCCESSFULLY!")
