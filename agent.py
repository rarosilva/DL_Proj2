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
import torch.nn as nn
import os
from collections import deque

N_ACTIONS = 2

class DQN(nn.Module):
    def __init__(self, n_frames = 1, dropout = 0.0):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3 * n_frames, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d((2, 2)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(32 * 2 * 2, N_ACTIONS)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

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

        self.n_frames = 4
        self.frame_buffer = deque(maxlen=self.n_frames)
        self.is_first_obs = True

        self.model = DQN(n_frames=4, dropout=0.0)
        model_path = os.path.join(os.path.dirname(__file__), "model.pt")
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

    def select_action(self, obs: np.ndarray) -> int:
        """
        Choose an action given the current observation.

        Args:
            obs: np.ndarray of shape (H, W, 3), dtype float32.

        Returns:
            action: int — 0 (move up) or 1 (move down).
        """

        #frame stacking
        if self.is_first_obs:
            for _ in range(self.n_frames):
                self.frame_buffer.append(obs.copy())
            self.is_first_obs = False
        else:
            self.frame_buffer.append(obs.copy())

        stacked_obs_np = np.concatenate(list(self.frame_buffer), axis=-1)


        obs = torch.tensor(stacked_obs_np.astype(np.float32) / 255.0)
        if obs.ndim == 3: # add batch dimension = 1 if a single frame is passed
            obs = obs.unsqueeze(0)
        obs = obs.permute(0, 3, 1, 2) # (N, C, H, W)
        
        with torch.no_grad():
            q_values = self.model(obs)
            action = q_values.argmax().item()
        return int(action)
