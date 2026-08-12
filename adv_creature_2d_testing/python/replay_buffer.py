import random
import torch
from typing import List, Dict, Tuple, Any

class ReplayBuffer:
    """
    Experience Replay Buffer for population-level M2 (action-selector) 
    and M3 (dynamics predictor) gradient stability (§6).
    """
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: List[Dict[str, Any]] = []
        self.position = 0

    def push(
        self,
        joint_ids: List[int],
        joint_angles: List[float],
        goal: torch.Tensor,
        action: torch.Tensor,
        real_delta_pos: torch.Tensor
    ):
        transition = {
            "joint_ids": joint_ids,
            "joint_angles": joint_angles,
            "goal": goal.detach().cpu(),
            "action": action.detach().cpu(),
            "real_delta_pos": real_delta_pos.detach().cpu()
        }
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)
