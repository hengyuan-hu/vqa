import torch
import torch.nn as nn
from gated_tanh import GatedTanh


class SimpleClassifier(object):
    def __init__(self, in_dim, hid_dim, out_dim):
        layers = [GatedTanh(in_dim, hid_dim), nn.Linear(hid_dim, out_dim)]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return nn.functional.sigmoid(logits)

    def loss(self, x, y):
        pass
