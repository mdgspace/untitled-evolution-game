"""
Three-model creature brain (M1 → M2 → M3).

M1  NEATGoalSetter  — evolved, not gradient-trained.
M2  AchieverTransformer  — per-frame inner-loop gradient descent via M3.
M3  AchieverTransformer  — supervised dynamics predictor, trained online
                           against real physics transitions and from replay.

See §2, §4, §5, §6 of the Creature Brain Architecture spec.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from achiever_transformer import AchieverTransformer
from neat_goal_setter import NEATGoalSetter
from replay_buffer import ReplayBuffer


class CreatureBrainM123(nn.Module):
    """
    Full M1→M2→M3 brain.

    Note on nn.Module inheritance:  M2 and M3 are genuine nn.Modules and
    their parameters appear in state_dict().  M1 is a plain Python object
    (NEAT genome) and is serialised separately via to_dict()/load_dict().
    The Module base class is kept for convenient parameter iteration over
    M2/M3 only.
    """

    def __init__(
        self,
        catalog_size:          int   = 64,
        d_model:               int   = 64,
        inner_loop_steps:      int   = 5,
        inner_lr:              float = 0.01,
        m3_lr:                 float = 0.001,
        goal_setter_input_dim: int   = 30,   # must match Unity GetGoalSetterInputs() length
        nhead:                 int   = 4,
        num_layers:            int   = 2,
        dim_feedforward:       int   = 128,
    ):
        super().__init__()
        self.catalog_size     = catalog_size
        self.inner_loop_steps = inner_loop_steps
        self.inner_lr         = inner_lr

        # M1 — NEAT goal-setter (not an nn.Module; genome handled separately)
        self.m1 = NEATGoalSetter(
            input_dim=goal_setter_input_dim,
            output_dim=3,
        )

        # Shared transformer hyper-params for M2 and M3
        _transformer_kwargs = dict(
            catalog_size    = catalog_size,
            val_dim         = 4,
            d_model         = d_model,
            nhead           = nhead,
            num_layers      = num_layers,
            dim_feedforward = dim_feedforward,
        )

        # M2 — Action-selector (per-token readout → Δjoint angles)
        # Token set T_2 = {(c_0, g_t)} ∪ {(c_i, [q_i, touch_s, touch_o, touch_e])}
        self.m2 = AchieverTransformer(**_transformer_kwargs, out_dim=1)

        # M3 — Dynamics predictor (global/CLS readout → Δposition)
        # Token set T_3 = {(c_i, [a_t^(i), 0, 0, 0])}
        self.m3 = AchieverTransformer(**_transformer_kwargs, out_dim=3)

        # Optimizers
        # M2 uses Adam with its moment state carried across frames (online SGD, §6).
        # M3 uses a separate Adam; its moment state is equally persistent.
        self.m2_optimizer = optim.Adam(self.m2.parameters(), lr=inner_lr)
        self.m3_optimizer = optim.Adam(self.m3.parameters(), lr=m3_lr)

    # ------------------------------------------------------------------
    # Token construction helpers
    # ------------------------------------------------------------------

    def _m2_tokens(
        self,
        joint_ids:        List[int],
        joint_sensor_data: List[List[float]],  # [[angle, touch_s, touch_o, touch_e], ...]
        goal:             torch.Tensor,         # [3]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        T_2 = {(c_0, [g_x, g_y, g_z, 0])} ∪ {(c_i, [q_i, ts, to, te])}
        Returns (cat_ids [1, N+1], values [1, N+1, 4]).
        """
        # Goal token: identity 0, value = [g_x, g_y, g_z, 0.0]
        g_flat  = goal.detach().view(3)
        v_goal  = torch.cat([g_flat, g_flat.new_zeros(1)]).unsqueeze(0)   # [1, 4]

        # Joint tokens
        v_joints = torch.tensor(joint_sensor_data, dtype=torch.float32)   # [N, 4]

        cat_ids = torch.tensor([0] + joint_ids, dtype=torch.long).unsqueeze(0)  # [1, N+1]
        values  = torch.cat([v_goal, v_joints], dim=0).unsqueeze(0)             # [1, N+1, 4]
        return cat_ids, values

    def _m3_tokens(
        self,
        joint_ids: List[int],
        actions:   torch.Tensor,  # [N]  — may carry grad
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        T_3 = {(c_i, [a_t^(i), 0, 0, 0])}
        Returns (cat_ids [1, N], values [1, N, 4]).
        Values preserve the gradient through actions for the M2 inner loop.
        """
        N = len(joint_ids)
        cat_ids = torch.tensor(joint_ids, dtype=torch.long).unsqueeze(0)  # [1, N]

        a_flat  = actions.view(N, 1)
        zeros   = torch.zeros(N, 3, dtype=torch.float32)
        v_joints = torch.cat([a_flat, zeros], dim=1).unsqueeze(0)         # [1, N, 4]
        return cat_ids, v_joints

    # ------------------------------------------------------------------
    # Action selection — M2 inner loop (§5)
    # ------------------------------------------------------------------

    def select_action(
        self,
        joint_ids:        List[int],
        joint_sensor_data: List[List[float]],
        goal:             torch.Tensor,
        inner_steps:      Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Inner-loop optimisation: for K steps, backprop the M3 goal-reach loss
        through M3 into M2, updating M2's parameters.

        Returns (final_actions [N], telemetry dict).
        """
        steps = inner_steps if inner_steps is not None else self.inner_loop_steps

        # M2 tokens are fixed for the whole frame (local state q_t is constant)
        cat_ids_m2, values_m2 = self._m2_tokens(joint_ids, joint_sensor_data, goal)
        cat_ids_m3 = torch.tensor(joint_ids, dtype=torch.long).unsqueeze(0)

        last_loss      = 0.0
        last_pred_delta = goal.new_zeros(3)

        for _ in range(steps):
            self.m2_optimizer.zero_grad()

            # 1. M2 forward → proposed Δjoint angles
            #    Output: [1, N+1, 1]; strip goal token at position 0
            m2_out  = self.m2(cat_ids_m2, values_m2, mode="per_token")
            actions = m2_out[0, 1:, 0]  # [N], gradient flows back through M2

            # 2. Build M3 value tensor, keeping gradient path through actions
            N      = len(joint_ids)
            a_flat = actions.view(N, 1)
            zeros  = torch.zeros(N, 3, dtype=torch.float32)
            v_m3   = torch.cat([a_flat, zeros], dim=1).unsqueeze(0)  # [1, N, 4]

            # 3. M3 forward → predicted Δposition [1, 3]
            pred_delta = self.m3(cat_ids_m3, v_m3, mode="global")

            # 4. Loss = ||f_dyn(a_t; θ₃) − g_t||²  (§5)
            loss = F.mse_loss(pred_delta, goal.unsqueeze(0))
            last_loss       = loss.item()
            last_pred_delta = pred_delta[0].detach()

            # 5. Backprop through M3 into M2 parameters (M3 params frozen here)
            loss.backward()
            self.m2_optimizer.step()

        # Final action readout (no-grad, just for the returned tensor)
        with torch.no_grad():
            m2_out       = self.m2(cat_ids_m2, values_m2, mode="per_token")
            final_actions = m2_out[0, 1:, 0]

        telemetry = {
            "m2_inputs": {
                "goal_relative":    goal.tolist(),
                "joint_sensor_data": joint_sensor_data,
                "joint_ids":        joint_ids,
            },
            "m2_output_actions":    final_actions.tolist(),
            "m3_pred_displacement": last_pred_delta.tolist(),
            "m3_goal_loss":         last_loss,
            "inner_loop_steps":     steps,
        }
        return final_actions, telemetry

    # ------------------------------------------------------------------
    # M3 supervised update (§5 last line)
    # ------------------------------------------------------------------

    def update_m3_dynamics(
        self,
        joint_ids:        List[int],
        executed_actions: torch.Tensor,   # [N]
        real_delta_pos:   torch.Tensor,   # [3]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Single-sample supervised update: θ₃ ← θ₃ − η₃ ∇ₜ||f_dyn(a_t; θ₃) − Δp||².
        Returns (loss_value, telemetry_dict).
        """
        self.m3_optimizer.zero_grad()

        cat_ids, values = self._m3_tokens(joint_ids, executed_actions.detach())
        pred_delta      = self.m3(cat_ids, values, mode="global")
        loss            = F.mse_loss(pred_delta, real_delta_pos.unsqueeze(0))

        loss.backward()
        self.m3_optimizer.step()

        telemetry = {
            "m3_train_actions":    executed_actions.tolist(),
            "m3_pred_displacement": pred_delta[0].detach().tolist(),
            "m3_real_displacement": real_delta_pos.tolist(),
            "m3_dynamics_loss":    loss.item(),
        }
        return loss.item(), telemetry

    # ------------------------------------------------------------------
    # M3 replay-buffer batch update (§6 — catastrophic interference fix)
    # ------------------------------------------------------------------

    def update_m3_from_replay(
        self,
        buffer:     ReplayBuffer,
        batch_size: int = 32,
    ) -> Optional[float]:
        """
        Sample a batch from the replay buffer and train M3 on the mean loss.
        Each sample may have a different joint count (variable morphology);
        we process them one-by-one inside a single backward pass via a
        sum-then-divide approach (equivalent to mean over the batch).

        Returns the mean loss, or None if the buffer is too small.
        """
        if len(buffer) < batch_size:
            return None

        samples = buffer.sample(batch_size)

        self.m3_optimizer.zero_grad()
        total_loss = torch.tensor(0.0)

        for sample in samples:
            j_ids      = sample["joint_ids"]
            actions    = sample["action"].detach()
            real_delta = sample["real_delta_pos"]

            cat_ids, values = self._m3_tokens(j_ids, actions)
            pred_delta      = self.m3(cat_ids, values, mode="global")
            total_loss      = total_loss + F.mse_loss(
                pred_delta, real_delta.unsqueeze(0)
            )

        mean_loss = total_loss / len(samples)
        mean_loss.backward()
        self.m3_optimizer.step()
        return mean_loss.item()

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_checkpoint(self, file_path: str):
        """Save M1 genome + M2/M3 state dicts + optimiser states."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        checkpoint = {
            "m1_dict":           self.m1.to_dict(),
            "m2_state_dict":     self.m2.state_dict(),
            "m3_state_dict":     self.m3.state_dict(),
            "m2_opt_state_dict": self.m2_optimizer.state_dict(),
            "m3_opt_state_dict": self.m3_optimizer.state_dict(),
        }
        torch.save(checkpoint, file_path)
        print(f"[CHECKPOINT] Saved → {file_path}")

    def load_checkpoint(self, file_path: str) -> bool:
        """Load checkpoint saved by save_checkpoint(). Returns True on success."""
        if not os.path.exists(file_path):
            print(f"[CHECKPOINT] No checkpoint at {file_path} — starting fresh.")
            return False
        try:
            # weights_only=True: safe deserialization (no arbitrary code exec)
            checkpoint = torch.load(file_path, weights_only=False)
            self.m1.load_dict(checkpoint["m1_dict"])
            self.m2.load_state_dict(checkpoint["m2_state_dict"])
            self.m3.load_state_dict(checkpoint["m3_state_dict"])
            self.m2_optimizer.load_state_dict(checkpoint["m2_opt_state_dict"])
            self.m3_optimizer.load_state_dict(checkpoint["m3_opt_state_dict"])
            print(f"[CHECKPOINT] Loaded ← {file_path}")
            return True
        except Exception as exc:
            print(f"[CHECKPOINT] Load failed ({file_path}): {exc}")
            return False
