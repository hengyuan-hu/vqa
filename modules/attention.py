import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
import numpy as np


class Attention(nn.Module):
    '''
    Generic attention modules that computes a soft attention distribution
    over a set of objects.
    '''

    def __init__(self, in_dim, hidden_dim, num_layers):
        super(Attention, self).__init__()

        layers = [weight_norm(nn.Linear(in_dim, hidden_dim), dim=None)]
        for i in range(1, num_layers):
            layers.append(weight_norm(hidden_dim, hidden_dim), dim=None)
            layers.append(ReLU())

        self.nonlinear = nn.Sequential(*layers)
        self.linear = weight_norm(nn.Linear(hidden_dim, 1), dim=None)

    def forward(self, inputs):
        '''
        inputs: [batch, num_objs, dim]

        return: [batch_size, num_objects]
        '''
        batch_size, num_objects, _ = inputs.size()
        inputs = inputs.view(batch_size * num_objects, -1)
        x = self.nonlinear(inputs)
        x = self.linear(x)
        x = x.view(batch_size, num_objects)
        weights = nn.functional.softmax(x)
        return weights

    def logits(self, inputs):
        """return the pre-softmax attention weights.
        inputs: [batch, num_objs, in_dim]

        return: [batch, num_objs]
        """
        batch_size, num_objects, _ = inputs.size()
        inputs = inputs.view(batch_size * num_objects, -1)
        x = self.nonlinear(inputs)
        x = self.linear(x)
        weights = x.view(batch_size, num_objects)
        return weights


class UniAttention(nn.Module):
    '''
    Generic attention modules that computes a soft attention distribution
    over a set of objects.
    '''

    def __init__(self, in_dim):
        super(UniAttention, self).__init__()

        self.linear = weight_norm(nn.Linear(in_dim, 1), dim=None)

    def forward(self, inputs):
        '''
        inputs: [batch, num_objs, dim]

        return: [batch_size, num_objects]
        '''
        x = self.linear(inputs).squeeze(2)
        weights = nn.functional.softmax(x)
        return weights


class NewAttention(nn.Module):
    '''
    Generic attention modules that computes a soft attention distribution
    over a set of objects.
    '''

    def __init__(self, v_dim, q_dim):
        super(NewAttention, self).__init__()

        self.v_proj = weight_norm(nn.Linear(v_dim, q_dim), dim=None)
        self.linear = weight_norm(nn.Linear(q_dim, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits)
        return w

    def logits(self, v, q):
        batch, k, _ = v.size()
        v_proj = self.v_proj(v) # [batch * k, qdim]
        q_expand = q.unsqueeze(1).repeat(1, k, 1)
        joint_repr = v_proj * q_expand
        joint_repr = nn.functional.normalize(joint_repr, 2, 2)

        logits = self.linear(joint_repr).view(batch, k)
        return logits
