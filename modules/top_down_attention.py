import torch
import torch.nn as nn
from attention import Attention


class TopDownAttention(Attention):
  '''
  Computes top down attention over a set of visual features given a
  question embedding.
  '''

  def __init__(self, q_dim, v_dim, hidden_dim):
    super(TopDownAttention, self).__init__(q_dim + v_dim, hidden_dim)

  def forward(self, v, q):
    '''
    v      Visual features
           Dimensions: batch_size x num_features x v_dim

    q      Question embedding
           Dimensions: batch_size x q_dim
    '''
    q = q.unsqueeze(1)
    q = q.expand(q.size(0), v.size(1), q.size(2))
    inputs = torch.cat((v, q), 2)

    return super(TopDownAttention, self).forward(inputs, v)
