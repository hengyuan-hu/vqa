import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from gated_tanh import GatedTanh


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(SimpleClassifier, self).__init__()
        layers = [
            GatedTanh(in_dim, hid_dim),
            nn.Dropout(0.5, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return nn.functional.sigmoid(logits)

    def loss(self, x, y):
        logits = self.main(x)
        l = nn.functional.binary_cross_entropy_with_logits(logits, y)
        l *= y.size(1)
        return l
