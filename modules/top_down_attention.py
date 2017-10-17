import torch
import torch.nn as nn
from attention import Attention

class TopDownAttention(Attention):
  '''
  Computes top down attention over a set of visual features given a
  question embedding.
  '''

  def __init__(self, num_features, q_len=512, v_len=2048, hidden_len=512):
    super(TopDownAttention, self).__init__(num_features, q_len + v_len, hidden_len)


  def forward(self, v, q):
    '''
    v      Visual features
           Dimensions: batch_size x num_features x v_len

    q      Question embedding
           Dimensions: batch_size x q_len
    '''
    q = q.unsqueeze(1)
    q = q.expand(q.size(0), v.size(1), q.size(2))
    inputs = torch.cat((v, q), 2)

    return super(TopDownAttention, self).forward(inputs, v)

