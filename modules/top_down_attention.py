import torch
from attention import Attention, SortAttention


class TopDownAttention(Attention):
  '''
  Computes top down attention over a set of visual features given a
  question embedding.
  '''

  def __init__(self, q_dim, v_dim, hidden_dim, num_layers=1):
    super(TopDownAttention, self).__init__(q_dim + v_dim, hidden_dim, num_layers)
    self.q_dim = q_dim

  def forward(self, v, q):
    '''
    v      Visual features
           Dimensions: batch_size x num_features x v_dim

    q      Question embedding
           Dimensions: batch_size x q_dim
    '''
    batch, num_feat, _ = v.size()
    q = q.unsqueeze(1).expand(batch, num_feat, self.q_dim)
    inputs = torch.cat((v, q), 2)

    return super(TopDownAttention, self).forward(inputs, v)

  def logits(self, v, q):
    batch, num_feat, _ = v.size()
    q = q.unsqueeze(1).expand(batch, num_feat, self.q_dim)
    inputs = torch.cat((v, q), 2)

    return super(TopDownAttention, self).logits(inputs)


class SortTopDownAttention(SortAttention):
  '''
  Computes top down attention over a set of visual features given a
  question embedding.
  '''

  def __init__(self, q_dim, v_dim, hidden_dim, num_layers=1):
    super(SortTopDownAttention, self).__init__(q_dim + v_dim, hidden_dim, num_layers)
    self.q_dim = q_dim

  def forward(self, v, q):
    '''
    v      Visual features
           Dimensions: batch_size x num_features x v_dim

    q      Question embedding
           Dimensions: batch_size x q_dim
    '''
    batch, num_feat, _ = v.size()
    q = q.unsqueeze(1).expand(batch, num_feat, self.q_dim)
    inputs = torch.cat((v, q), 2)

    return super(SortTopDownAttention, self).forward(inputs, v)

  def logits(self, v, q):
    batch, num_feat, _ = v.size()
    q = q.unsqueeze(1).expand(batch, num_feat, self.q_dim)
    inputs = torch.cat((v, q), 2)

    return super(SortTopDownAttention, self).logits(inputs)
