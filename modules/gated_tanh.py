import torch.nn as nn


class GatedTanh(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(GatedTanh, self).__init__()

    self.sigmoid_linear = nn.Linear(in_dim, out_dim)
    self.tanh_linear = nn.Linear(in_dim, out_dim)

  def forward(self, x):
    out = nn.functional.relu(self.tanh_linear(x))
    # out = nn.functional.elu(self.tanh_linear(x))
    # out = nn.functional.tanh(self.tanh_linear(x))

    # tanh = nn.functional.tanh(self.tanh_linear(x))
    # sigmoid = nn.functional.sigmoid(self.sigmoid_linear(x))
    # out = tanh * sigmoid
    return out
