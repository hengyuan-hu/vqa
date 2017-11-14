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


class DualClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(DualClassifier, self).__init__()
        self.cls1 = self._build_network()
        self.cls2 = self._build_network()

    def _build_network(self):
        layers = [
            weight_norm(nn.Linear(hid_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(0.5, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        logits1 = self.cls1(x1)
        logits2 = self.cls2(x2)
        logits = torch.max(logits1, logits2)
        return logits
