import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from glu import GLU


_FLOAT32_MAX = np.finfo(np.float32).max
_HALF_LOG_MAX = float(np.log(_FLOAT32_MAX) / 2)


def log_sum_exp(logs, dim, keepdim):
    """Compute the log(sum(exp(x))) along axis specified by dim.

    NOTE: this won't reduce any dim
    """

    alpha = logs.max(dim, keepdim=True)[0] - _HALF_LOG_MAX
    logs = logs - alpha.expand_as(logs)
    if not keepdim:
        alpha = alpha.squeeze(dim)
    ls = alpha + torch.log(torch.sum(torch.exp(logs), dim, keepdim))
    return ls


def softmax(x, dim):
    a = x.max(dim, keepdim=True)[0] - _HALF_LOG_MAX
    x = x - a.expand_as(x)
    exp_x = torch.exp(x)
    sum_exp = exp_x.sum(dim, keepdim=True).expand_as(exp_x)
    return exp_x / sum_exp


def compute_relative_attention(w_logits, batch, num_objs):
    # w_logits: [batch, num_objs, num_objs] row repeat
    a = w_logits.max() - _HALF_LOG_MAX
    w_logits = w_logits - a#.expand_as(w_logits)
    exp_w = torch.exp(w_logits)

    # remove diagnal
    neg_identity = 1 - torch.eye(num_objs, num_objs).cuda()
    neg_identity = neg_identity.unsqueeze(0).expand(batch, num_objs, num_objs)
    exp_w = exp_w * Variable(neg_identity)

    w = exp_w / exp_w.sum(2, keepdim=True)
    return w


class RAM(nn.Module):
    def __init__(self, v_att, r_net):
        super(RAM, self).__init__()
        self.v_att = v_att
        self.r_net = r_net

    def forward2(self, v, q):
        """
        v: [batch, num_objs, v_dim]
        q: [batch, q_dim]
        """
        batch, num_objs, v_dim = v.size()

        w_logits = self.v_att.logits(v, q)
        # w_logits: [batch, num_objs]
        w_logits = w_logits.unsqueeze(1).expand(batch, num_objs, num_objs)
        # w_logits: [batch, num_objs, num_objs] row repeat
        w = compute_relative_attention(w_logits, batch, num_objs)
        w = w.unsqueeze(3)

        v = v.unsqueeze(1).expand(batch, num_objs, num_objs, v_dim)
        v = (v * w).sum(2)
        return v

    def forward(self, v, q):
        # print 'in ram forward: v', v.mean().data[0]
        # print 'in ram forward: q', q.mean().data[0]
        batch, num_objs, v_dim = v.size()
        # w_logits = self.v_att.logits(v, q)
        # norms, w = compute_ratio(w_logits)#.unsqueeze(2)
        # norms = norms.unsqueeze(2)
        # w = w.unsqueeze(2)
        # # print w.size()
        # # print v.size()
        # # w = nn.functional.softmax(w_logits).unsqueeze(2)

        # weighted_v = w * v
        # sum_v = weighted_v.sum(1, keepdim=True)
        # norms += 1
        # # print norms.min()
        # v_neg = (sum_v - weighted_v) / norms

        v_neg = self.forward2(v, q)

        q = q.unsqueeze(1).repeat(1, num_objs, 1)
        # TODO: use weighted_v ?
        vq_pairs = torch.cat([v, v_neg, q], 2)
        # print '>>>', vq_pairs.size()
        vq_pairs = vq_pairs.view(batch * num_objs, -1)
        out = self.r_net(vq_pairs)
        out = out.view(batch, num_objs, -1)
        return out


def compute_ratio(w_logits):
    """ the ratio to scale the v_{-} after subtracting v_{i}

    w_logits : [batch, num_objs]
    return   : [batch, num_objs]
    """
    # print w_logits.size()
    # print w_logits.mean()
    # print w_logits.max(1)
    a = w_logits.max() - _HALF_LOG_MAX
    w_logits = w_logits - a
    exp_w = torch.exp(w_logits)

    sum_exp = exp_w.sum(1, keepdim=True)
    partial_sum_exp = sum_exp - exp_w
    # sum_exp = Variable(sum_exp.data)
    # ratio = sum_exp / partial_sum_exp
    # print ratio.mean()
    # w = exp_w # / sum_exp
    return partial_sum_exp, exp_w


if __name__ == '__main__':
    import time
    from top_down_attention import TopDownAttention

    v = torch.rand(512, 36, 2048).cuda()
    q = torch.rand(512, 512).cuda()
    v_att = TopDownAttention(512, 2048, 512)
    r_net = nn.Sequential(
        GLU(2048 * 2 + 512, 2048),
        GLU(2048, 2048)
        # GLU(2048, 512),
    )

    # v = torch.rand(1, 50, 20).cuda()
    # q = torch.rand(1, 20).cuda()
    # v_att = TopDownAttention(20, 20, 2).cuda()

    v = Variable(v)
    q = Variable(q)

    arn = RAM(v_att, r_net).cuda()
    v1 = arn.forward(v, q)
    print v1.size()
    # v2 = arn.forward2(v, q)

    # # print v1[0]
    # # print v2[0]

    # print (v1 - v2).max()

    # t = time.time()
    # for i in range(100):
    #     arn.forward(v, q)
    #     # break

    # print 'time:', time.time() - t

    # t = time.time()
    # for i in range(100):
    #     arn.forward2(v, q)
    #     # break

    # print 'time:', time.time() - t
