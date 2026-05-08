import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, n_actions = 2, n_frames = 1):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3*n_frames, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d((2, 2)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(32 * 2 * 2, n_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x