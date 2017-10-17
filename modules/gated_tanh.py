import torch.nn as nn

class GatedTanh(nn.Module):

  def __init__(self, in_dim, out_dim):
    super(GatedTanh, self).__init__()

    self.sigmoid_linear = nn.Linear(in_dim, out_dim)
    self.sigmoid = nn.Sigmoid()

    self.tanh_linear = nn.Linear(in_dim, out_dim)
    self.tanh = nn.Tanh()

  def forward(self, x):
    tanh = self.tanh(self.tanh_linear(x))
    sigmoid = self.sigmoid(self.sigmoid_linear(x))
    out = tanh * sigmoid
    return out

