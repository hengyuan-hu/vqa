import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from glu import GLU
import numpy as np


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


class NewAttention(nn.Module):
    '''
    Generic attention modules that computes a soft attention distribution
    over a set of objects.
    '''

    def __init__(self, v_dim, q_dim):
        super(NewAttention, self).__init__()

        self.v_proj = GLU(v_dim, q_dim)
        self.linear = weight_norm(nn.Linear(q_dim, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        batch, k, vdim = v.size()
        # v_expand = v.view(batch * k, vdim)
        # print v.size()
        v_proj = self.v_proj(v) # [batch * k, qdim]
        # print v_proj.size()
        q_expand = q.unsqueeze(1).repeat(1, k, 1)#.view(batch * k, -1)
        # print v_proj.size(), q_expand.size()
        joint_repr = v_proj * q_expand

        # print 'where?'
        logits = self.linear(joint_repr).view(batch, k)
        w = nn.functional.softmax(logits).unsqueeze(2).expand_as(v)
        # print w.size(), v.size()
        out = w * v
        # print 'warinig?'
        return out
