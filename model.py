import torch.nn as nn

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

class BiggerDQN(nn.Module):
    def __init__(self, n_frames=1):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3 * n_frames, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.01),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.01),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.01),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, N_ACTIONS)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class GroupsDQN(nn.Module):
    def __init__(self, n_frames=1):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3 * n_frames, 32 * n_frames, kernel_size=3, stride=1, padding=1, groups=n_frames),
            nn.ReLU(),

            nn.Conv2d(32 * n_frames, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, N_ACTIONS)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x