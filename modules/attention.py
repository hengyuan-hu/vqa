import torch.nn as nn
from gated_tanh import GatedTanh

class Attention(nn.Module):
  '''
  Generic attention modules that computes a soft attention distribution
  over a set of objects.
  '''

  def __init__(self, num_objects, in_len, hidden_len):
    super(Attention, self).__init__()

    self.nonlinear = GatedTanh(in_len, hidden_len)
    self.linear = nn.Linear(hidden_len, 1)
    self.softmax = nn.Softmax()


  def forward(self, inputs, objects):
    '''
    inputs    Input vector x_i is used to compute weight a_i for object o_i.
              Dimensions: batch_size x num_objects x in_len

    objects   Final output is sum(a_i * o_i) for all objects o_i
              Dimensions: batch_size x num_objects x obj_len
    '''

    batch_size = inputs.size(0)
    num_objects = inputs.size(1)
    in_len = inputs.size(2)

    inputs = inputs.view(batch_size * num_objects, -1)
    x = self.nonlinear(inputs)
    x = self.linear(x)

    x = x.view(-1, num_objects)
    weights = self.softmax(x)
    weights = weights.unsqueeze(2)
    weights = weights.expand(weights.size(0), weights.size(1), objects.size(2))

    out = objects * weights
    return out

