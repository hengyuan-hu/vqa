import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from glu import GLU


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(SimpleClassifier, self).__init__()
        layers = [
            GLU(in_dim, hid_dim),
            nn.Dropout(0.5, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits
