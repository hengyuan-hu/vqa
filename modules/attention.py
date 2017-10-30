import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from glu import GLU
import numpy as np


_FLOAT32_MAX = np.finfo(np.float32).max
_HALF_LOG_MAX = float(np.log(_FLOAT32_MAX) / 2)


def softmax(x, dim):
    a = x.max(dim, keepdim=True)[0] - _HALF_LOG_MAX
    x = x - a.expand_as(x)
    exp_x = torch.exp(x)
    sum_exp = exp_x.sum(dim, keepdim=True).expand_as(exp_x)
    return exp_x / sum_exp



class Attention(nn.Module):
  '''
  Generic attention modules that computes a soft attention distribution
  over a set of objects.
  '''

  def __init__(self, in_dim, hidden_dim, num_layers):
    super(Attention, self).__init__()

    layers = [GLU(in_dim, hidden_dim)]
    for i in range(1, num_layers):
      layers.append(GLU(hidden_dim, hidden_dim))

    self.nonlinear = nn.Sequential(*layers)
    self.linear = weight_norm(nn.Linear(hidden_dim, 1), dim=None)

  def forward(self, inputs, objects):
    '''
    inputs    Input vector x_i is used to compute weight a_i for object o_i.
              Dimensions: batch_size x num_objects x in_dim

    objects   Final output is sum(a_i * o_i) for all objects o_i
              Dimensions: batch_size x num_objects x obj_dim

    return    [batch_size, num_objects, obj_dim]
    '''
    # assert inputs.size()[:2] == objects.size()[:2]
    batch_size, num_objects, _ = inputs.size()

    inputs = inputs.view(batch_size * num_objects, -1)
    x = self.nonlinear(inputs)
    x = self.linear(x)
    x = x.view(batch_size, num_objects)

    weights = nn.functional.softmax(x)
    weights = weights.unsqueeze(2).expand_as(objects)

    out = objects * weights
    return out

  def logits(self, inputs):
    """return the pre-softmax attention weights.

    inputs: [batch, num_objs, in_dim]

    return: [batch, num_objs]
    """
    batch_size, num_objects, _ = inputs.size()

    inputs = inputs.view(batch_size * num_objects, -1)
    x = self.nonlinear(inputs)
    x = self.linear(x)
    x = x.view(batch_size, num_objects)
    return x


class CrossAttention(nn.Module):
  def __init__(self, q_dim, v_dim, hidden_dim, num_layers):
    super(CrossAttention, self).__init__()

    in_dim = q_dim + v_dim
    layers = [GLU(in_dim, hidden_dim)]
    for i in range(1, num_layers):
      layers.append(GLU(hidden_dim, hidden_dim))

    self.nonlinear = nn.Sequential(*layers)
    self.linear = weight_norm(nn.Linear(hidden_dim, 1), dim=None)

  def forward(self, v, q):
    """
    v: [batch, k, vdim]
    q: [batch, l, qdim]
    """

    # print q.size()
    batch, l, qdim = q.size()
    _, k, vdim = v.size()

    v_expand = v.unsqueeze(1).repeat(1, l, 1, 1).view(batch*k*l, vdim)
    q_expand = q.unsqueeze(2).repeat(1, 1, k, 1).view(batch*l*k, qdim)
    vq = torch.cat([v_expand, q_expand], 1)

    logits = self.linear(self.nonlinear(vq))
    logits = logits.view(batch, l, k)
    w = softmax(logits, 2)

    # print w.size(), v_expand.size()
    v_att = v.unsqueeze(1) * w.unsqueeze(3)
    # print v.unsqueeze(1).size(), w.unsqueeze(3).size()
    v_att = v_att.sum(2)
    return v_att


if __name__ == '__main__':
  import torch
  from torch.autograd import Variable

  inputs = torch.rand(10, 3, 1024)
  objects = torch.rand(10, 3, 512)

  attention = Attention(1024, 512)
  out = attention.forward(Variable(inputs), Variable(objects))
  print out.size()
