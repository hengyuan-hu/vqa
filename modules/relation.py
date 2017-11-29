import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from torch.autograd import Variable
# from glu import GLU
import numpy as np
from fc import FCNet


_FLOAT32_MAX = np.finfo(np.float32).max
_HALF_LOG_MAX = float(np.log(_FLOAT32_MAX) / 2)


def softmax(x, dim):
    a = x.max(dim, keepdim=True)[0] - _HALF_LOG_MAX
    x = x - a.expand_as(x)
    exp_x = torch.exp(x)
    sum_exp = exp_x.sum(dim, keepdim=True).expand_as(exp_x)
    return exp_x / sum_exp


class RelationModule(nn.Module):
    def __init__(self, s_dim, q_dim, num_hid):
        super(RelationModule, self).__init__()

        self.v_conv = nn.Sequential(
            weight_norm(nn.Linear(2, 32), dim=None),
            nn.ReLU())
        self.v_proj = nn.Sequential(
            weight_norm(nn.Linear(128, 128), dim=None),
            nn.ReLU())

        self.q_proj = nn.Sequential(
            weight_norm(nn.Linear(q_dim, 128), dim=None),
            nn.ReLU())

        self.dropout = nn.Dropout(0.5)
        self.linear = weight_norm(nn.Linear(128, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim (or spatial_dim)]
        q: [batch, qdim]
        """
        batch, k, _ = v.size()

        vi = v.unsqueeze(1).repeat(1, k, 1, 1)
        vj = v.unsqueeze(2).repeat(1, 1, k, 1)
        v_stack = torch.stack([vi, vj], 4) # [batch, k, k, 4, 2]
        v_conv = self.v_conv(v_stack) # [batch, k, k, 4, 32]
        v_conv = v_conv.view(batch, k, k, -1) # [batch, k, k, 128]
        v_proj = self.v_proj(v_conv) # [batch, k, k, hid]

        q_proj = self.q_proj(q) # [batch, hid]
        q_proj = q_proj.unsqueeze(1).unsqueeze(1).expand_as(v_proj)
        # q_proj [batch, k, k, hid]
        # assert q_proj.size() == q_proj.size()

        joint_repr = v_proj * q_proj
        joint_repr = self.dropout(joint_repr)
        # joint_repr = nn.functional.normalize(joint_repr, 2, 3)
        logits = self.linear(joint_repr).squeeze(3) # [batch, k, k]
        return logits
        # w = softmax(logits, 1)
        # return w


class RelationAttention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid):
        super(RelationAttention, self).__init__()

        self.v_proj = FCNet([v_dim, num_hid], 0)
        self.q_proj = FCNet([q_dim, num_hid], 0)
        self.num_hid = num_hid

        # self.q_proj = nn.Sequential(
        #     weight_norm(nn.Linear(q_dim, 128), dim=None),
        #     nn.ReLU())

        self.dropout = nn.Dropout(0.2)
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim (or spatial_dim)]
        q: [batch, qdim]
        """
        batch, k, v_dim = v.size()

        v_proj = self.v_proj(v)
        vi = v_proj.unsqueeze(1).expand(batch, k, k, self.num_hid)
        vj = v_proj.unsqueeze(2).expand(batch, k, k, self.num_hid)
        v_proj = vi + vj # [batch, k, k, num_hid]

        q_proj = self.q_proj(q) # [batch, hid]
        q_proj = q_proj.unsqueeze(1).unsqueeze(1).expand_as(v_proj)
        # q_proj [batch, k, k, hid]
        # assert q_proj.size() == q_proj.size()

        joint_repr = v_proj * q_proj
        joint_repr = self.dropout(joint_repr)
        # joint_repr = nn.functional.normalize(joint_repr, 2, 3)
        logits = self.linear(joint_repr).squeeze(3) # [batch, k, k]
        return logits
        # w = softmax(logits, 1)
        # return w


if __name__ == '__main__':
    rm = RelationModule(4, 5, 2)

    v = Variable(torch.rand(2, 3, 4))
    q = Variable(torch.rand(2, 5))

    w = rm(v, q)

    att = Variable(torch.rand(2, 3)) * 100
    att = nn.functional.softmax(att)
    att = att.unsqueeze(2)
    print 'att:', att
    print 'w:', w

    prop_att = torch.bmm(w, att).squeeze()
    print prop_att
    print prop_att.sum(1)
