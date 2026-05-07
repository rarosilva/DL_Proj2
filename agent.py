"""
SpaceRace — Agent Submission Template
======================================
This is the ONLY file you need to submit to Codabench.

Rules:
  - Keep the class name exactly as `Agent`.
  - Keep the method signature of `select_action` unchanged.
  - You may add imports, helper methods, and load model weights in __init__.
  - Do NOT import pygame or call env.render() — the eval worker is headless.

Observation layout (shape = H x W x 3, dtype float32):
  channel 0  — ship position  (1.0 at ship cell, 0.0 elsewhere)
  channel 1  — debris cells   (1.0 where debris occupies, 0.0 elsewhere)
  channel 2  — time remaining (uniform plane, value in [0, 1])

Actions:
  0 → move up
  1 → move down
"""

from __future__ import annotations
import numpy as np
import torch
from model import DQN

class Agent:
    """Your RL agent for SpaceRace-v0."""

    def __init__(self) -> None:
        """
        Initialize your agent here.

        Examples of what you might do:
          - Load a saved model:   self.model = torch.load("model.pt")
          - Set up any state your agent needs between steps.

        If you load a file, make sure it is included in your submission zip.
        """
        self.model = DQN()
        self.model.load_state_dict(torch.load("model.pt"))

    def select_action(self, obs: np.ndarray) -> int:
        """
        Choose an action given the current observation.

        Args:
            obs: np.ndarray of shape (H, W, 3), dtype float32.

        Returns:
            action: int — 0 (move up) or 1 (move down).
        """
        t = torch.tensor(obs, dtype=torch.float32)
        t = t / 255.0
        t = t.permute(2, 0, 1)
        obs = t.unsqueeze(0)

        with torch.no_grad():
            q_values = self.model(obs)
            action = q_values.argmax(dim=1).item()
        return int(action)
