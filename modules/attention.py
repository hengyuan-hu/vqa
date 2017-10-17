import torch.nn as nn
from gated_tanh import GatedTanh


class Attention(nn.Module):
  '''
  Generic attention modules that computes a soft attention distribution
  over a set of objects.
  '''

  def __init__(self, num_objects, in_dim, hidden_dim):
    super(Attention, self).__init__()

    self.nonlinear = GatedTanh(in_dim, hidden_dim)
    self.linear = nn.Linear(hidden_dim, 1)

  def forward(self, inputs, objects):
    '''
    inputs    Input vector x_i is used to compute weight a_i for object o_i.
              Dimensions: batch_size x num_objects x in_dim

    objects   Final output is sum(a_i * o_i) for all objects o_i
              Dimensions: batch_size x num_objects x obj_dim
    '''
    assert inputs.size()[:2] == objects.size()[:2]
    batch_size, num_objects, _ = inputs.size()

    inputs = inputs.view(batch_size * num_objects, -1)
    x = self.nonlinear(inputs)
    x = self.linear(x)
    x = x.view(batch_size, num_objects)

    weights = nn.functional.softmax(x)
    weights = weights.unsqueeze(2).expand_as(objects)

    out = objects * weights
    return out
