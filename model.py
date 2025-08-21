import torch.nn as nn

class MLPScorer(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: (batch, seq_len, in_dim)
        return self.fc(x)  # (batch, seq_len, 1)
