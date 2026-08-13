"""The MLP used for maternal health risk classification.

Reused as-is by the FL client (step 5) - every simulated hospital trains
its own instance of this same architecture on its own local data shard.
"""

import torch.nn as nn


class MaternalRiskMLP(nn.Module):
    """6 vitals in -> Low/Mid/High risk logits out.

    No Softmax layer: nn.CrossEntropyLoss applies log_softmax internally,
    so the model outputs raw logits and softmax is only applied (if ever
    needed) at inference time via torch.softmax(logits, dim=1).
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        return self.net(x)
