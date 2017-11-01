import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.nn.utils.weight_norm import weight_norm
from relational import RelationalNet
from glu import GLU

class RelationalAttention(nn.Module):

    def __init__(self, v_len, q_len, h=1024, r_layers=2, pair_layers=1, num_objects=36):
        super(RelationalAttention, self).__init__()

        self.RN = RelationalNet(v_len, q_len, 1, h=h, num_layers=r_layers)
        layers = [weight_norm(nn.Linear(2 * v_len + q_len, h)), GLU(h,h)]
        for i in xrange(pair_layers - 1):
            layers += [weight_norm(nn.Linear(h,h)), GLU(h,h)]

        #layers += [weight_norm(nn.Linear(h,h))]

        self.pair_net = nn.Sequential(*layers)

        self.mask = Variable(1 - torch.eye(num_objects), requires_grad=False).cuda()
        self.mask = self.mask.view(1, num_objects, num_objects, 1)

    def forward(self, v, q):
        batch_size, num_objects, vdim = v.size()

        weights = self.RN(v, q)

        # Use mask to remove self relations
        mask = self.mask.expand(batch_size, num_objects, num_objects, 1)
        weights = weights * mask

        # Normalize
        weights = weights.view(batch_size * num_objects, -1)
        weights = nn.functional.softmax(weights)

        # Compute attention sum
        weights = weights.view(batch_size, num_objects, num_objects, 1)
        weights = weights.expand(batch_size, num_objects, num_objects, vdim)
        v_others = v.unsqueeze(1)
        v_others = v_others.expand(batch_size, num_objects, num_objects, vdim)
        v_others = weights * v_others
        v_others = v_others.sum(2) # batch_size x num_objects x vdim

        # Compute relationship between v and v_others
        q = q.unsqueeze(1)
        q = q.expand(batch_size, num_objects, q.size(2))

        v_combined = torch.cat([v, v_others, q], dim=2)
        v_combined = v_combined.view(batch_size * num_objects, -1)
        out = self.pair_net.forward(v_combined)
        out = out.view(batch_size, num_objects, -1)

        return out

if __name__ == '__main__':
    import torch
    from torch.autograd import Variable

    v = torch.rand(10, 3, 1024)
    q = torch.rand(10, 512)
    rn = RelationalAttention(1024, 512)
    out = rn.forward(Variable(v), Variable(q))
    print out.size()
    print out

