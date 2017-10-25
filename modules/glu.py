import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm


class GLU(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(GLU, self).__init__()

    self.linear = weight_norm(nn.Linear(in_dim, out_dim), dim=None)
    self.gate = weight_norm(nn.Linear(in_dim, out_dim), dim=None)

  def forward(self, x):
    linear = self.linear(x)
    gate = nn.functional.sigmoid(self.gate(x))
    out = linear * gate
    return out
