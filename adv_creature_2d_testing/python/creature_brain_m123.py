import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
from achiever_transformer import AchieverTransformer
from neat_goal_setter import NEATGoalSetter

class CreatureBrainM123(nn.Module):
    """
    3-Model Creature Brain Architecture:
      - M1: NEAT Goal-Setter Network -> g_t (processes 5-frame memory buffer)
      - M2: Action Selector Transformer -> a_t (inputs: Goal + Joint Angles & Touch Sensors)
      - M3: Dynamics Predictor Transformer -> predicted delta position
    """
    def __init__(
        self,
        catalog_size: int = 64,
        d_model: int = 64,
        inner_loop_steps: int = 5,
        inner_lr: float = 0.01,
        m3_lr: float = 0.001,
        goal_setter_input_dim: int = 50
    ):
        super().__init__()
        self.catalog_size = catalog_size
        self.inner_loop_steps = inner_loop_steps
        self.inner_lr = inner_lr
        
        # M1 NEAT Goal-Setter Model
        self.m1 = NEATGoalSetter(input_dim=goal_setter_input_dim, output_dim=3)
        
        # M2 Achiever Transformer (Action Selector)
        # Inputs: Goal token (c=0, v=[g_x, g_y, g_z, 0]) + Joint tokens (c=c_i, v=[q_i, touch_s, touch_o, touch_e])
        # Readout mode: per_token -> scalar delta angle per joint
        self.m2 = AchieverTransformer(
            catalog_size=catalog_size,
            val_dim=4,
            d_model=d_model,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            out_dim=1
        )
        
        # M3 Achiever Transformer (Dynamics Predictor)
        # Inputs: Joint tokens (c=c_i, v=[a_i, 0, 0, 0])
        # Readout mode: global -> 3D predicted displacement delta position
        self.m3 = AchieverTransformer(
            catalog_size=catalog_size,
            val_dim=4,
            d_model=d_model,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            out_dim=3
        )
        
        # Optimizer for M3 (dynamics predictor updated against real physics transitions)
        self.m3_optimizer = optim.Adam(self.m3.parameters(), lr=m3_lr)
        # Optimizer for M2 online parameter adaptation
        self.m2_optimizer = optim.Adam(self.m2.parameters(), lr=inner_lr)

    def prepare_m2_tokens(
        self,
        joint_ids: List[int],
        joint_sensor_data: List[List[float]], # List of [angle, touch_self, touch_other, touch_env]
        goal: torch.Tensor # [3]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build Token Set T_2 = {(c_0, [g_x, g_y, g_z, 0])} U {(c_i, [q_i, touch_s, touch_o, touch_e])}
        """
        num_joints = len(joint_ids)
        cat_ids = torch.tensor([0] + joint_ids, dtype=torch.long) # [num_joints + 1]
        
        # Goal vector as value for token 0 (padded to 4D)
        g_flat = goal.detach().view(3)
        v_goal = torch.cat([g_flat, torch.tensor([0.0])]).unsqueeze(0) # [1, 4]
        
        # Joint values [angle, touch_self, touch_other, touch_env] (4D)
        v_joints = torch.tensor(joint_sensor_data, dtype=torch.float32) # [num_joints, 4]
        
        values = torch.cat([v_goal, v_joints], dim=0) # [num_joints + 1, 4]
        
        return cat_ids.unsqueeze(0), values.unsqueeze(0)

    def prepare_m3_tokens(
        self,
        joint_ids: List[int],
        actions: torch.Tensor # [num_joints]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build Token Set T_3 = {(c_i, [a_t^(i), 0, 0, 0])}
        """
        num_joints = len(joint_ids)
        cat_ids = torch.tensor(joint_ids, dtype=torch.long).unsqueeze(0) # [1, num_joints]
        
        a_flat = actions.view(-1, 1) # [num_joints, 1]
        zeros = torch.zeros((num_joints, 3), dtype=torch.float32)
        v_joints = torch.cat([a_flat, zeros], dim=1).unsqueeze(0) # [1, num_joints, 4]
        
        return cat_ids, v_joints

    def select_action(
        self,
        joint_ids: List[int],
        joint_sensor_data: List[List[float]],
        goal: torch.Tensor,
        inner_steps: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Runs M2 action optimization loop (§5):
        Per-frame inner loop backpropagating loss through M3 into M2.
        Returns (final_actions, m2_m3_telemetry).
        """
        steps = inner_steps if inner_steps is not None else self.inner_loop_steps
        cat_ids_m2, values_m2 = self.prepare_m2_tokens(joint_ids, joint_sensor_data, goal)
        cat_ids_m3 = torch.tensor(joint_ids, dtype=torch.long).unsqueeze(0)
        
        last_loss = 0.0
        last_pred_delta = torch.zeros(3)

        for step in range(steps):
            self.m2_optimizer.zero_grad()
            
            # 1. Forward M2 -> proposed joint deltas [1, num_tokens, 1]
            m2_out = self.m2(cat_ids_m2, values_m2, mode="per_token")
            actions = m2_out[:, 1:, 0]
            
            # 2. Build M3 input token values using M2's actions (keeping gradient flow!)
            num_joints = len(joint_ids)
            a_flat = actions.view(-1, 1) # [num_joints, 1]
            zeros = torch.zeros((num_joints, 3), dtype=torch.float32)
            v_m3 = torch.cat([a_flat, zeros], dim=1).unsqueeze(0) # [1, num_joints, 4]
            
            # 3. Forward M3 -> predicted delta position [1, 3]
            pred_delta_pos = self.m3(cat_ids_m3, v_m3, mode="global")
            
            # 4. Loss = || f_dyn(a_t; theta_3) - g_t ||^2
            loss = F.mse_loss(pred_delta_pos, goal.unsqueeze(0))
            last_loss = loss.item()
            last_pred_delta = pred_delta_pos[0].detach()
            
            # 5. Backprop through M3 into M2 parameters
            loss.backward()
            self.m2_optimizer.step()
            
        # Final pass after inner optimization steps
        with torch.no_grad():
            m2_out = self.m2(cat_ids_m2, values_m2, mode="per_token")
            final_actions = m2_out[0, 1:, 0]
            
        telemetry = {
            "m2_inputs": {
                "goal_relative": goal.tolist(),
                "joint_sensor_data": joint_sensor_data,
                "joint_ids": joint_ids
            },
            "m2_output_actions": final_actions.tolist(),
            "m3_pred_displacement": last_pred_delta.tolist(),
            "m3_goal_loss": last_loss,
            "inner_loop_steps": steps
        }
        return final_actions, telemetry

    def update_m3_dynamics(
        self,
        joint_ids: List[int],
        executed_actions: torch.Tensor,
        real_delta_pos: torch.Tensor
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Trains M3 via supervised regression against real outcomes from physics engine.
        Returns (loss_float, m3_train_telemetry).
        """
        self.m3_optimizer.zero_grad()
        cat_ids, values = self.prepare_m3_tokens(joint_ids, executed_actions)
        
        pred_delta_pos = self.m3(cat_ids, values, mode="global")
        loss = F.mse_loss(pred_delta_pos, real_delta_pos.unsqueeze(0))
        
        loss.backward()
        self.m3_optimizer.step()

        telemetry = {
            "m3_train_actions": executed_actions.tolist(),
            "m3_pred_displacement": pred_delta_pos[0].detach().tolist(),
            "m3_real_displacement": real_delta_pos.tolist(),
            "m3_dynamics_loss": loss.item()
        }
        return loss.item(), telemetry

    def save_checkpoint(self, file_path: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        checkpoint = {
            "m1_dict": self.m1.to_dict(),
            "m2_state_dict": self.m2.state_dict(),
            "m3_state_dict": self.m3.state_dict(),
            "m2_opt_state_dict": self.m2_optimizer.state_dict(),
            "m3_opt_state_dict": self.m3_optimizer.state_dict(),
        }
        torch.save(checkpoint, file_path)
        print(f"[CHECKPOINT] Saved brain model parameters to {file_path}")

    def load_checkpoint(self, file_path: str) -> bool:
        if not os.path.exists(file_path):
            print(f"[CHECKPOINT] No existing checkpoint found at {file_path}. Starting fresh.")
            return False
        try:
            checkpoint = torch.load(file_path)
            self.m1.load_dict(checkpoint["m1_dict"])
            self.m2.load_state_dict(checkpoint["m2_state_dict"])
            self.m3.load_state_dict(checkpoint["m3_state_dict"])
            self.m2_optimizer.load_state_dict(checkpoint["m2_opt_state_dict"])
            self.m3_optimizer.load_state_dict(checkpoint["m3_opt_state_dict"])
            print(f"[CHECKPOINT] Successfully loaded brain model parameters from {file_path}")
            return True
        except Exception as e:
            print(f"[CHECKPOINT] Error loading checkpoint {file_path}: {e}")
            return False
