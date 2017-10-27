import torch
import torch.nn as nn
from glu import GLU
from torch.nn.utils.weight_norm import weight_norm

class RelationalNet(nn.Module):

    def __init__(self, v_len, q_len, output_len, h=1024, num_layers=1):
        super(RelationalNet, self).__init__()

        layers = [weight_norm(nn.Linear(v_len * 2 + q_len, h)), GLU(h, h)]

        for i in xrange(num_layers - 1):
            layers += [weight_norm(nn.Linear(h, h)), GLU(h, h)]

        layers += [weight_norm(nn.Linear(h, output_len))]
        self.g = nn.Sequential(*layers)


    def forward(self, v, q):
        '''
        v    batch_size x num_objects x vdim
        q    batch_size x qdim

        returns    batch_size x num_objects x num_objects x output_len
        '''

        batch_size, num_objects, vdim = v.size()
        v1 = torch.unsqueeze(v, 1)
        v1 = v1.expand(batch_size, num_objects, num_objects, vdim)
        v2 = torch.unsqueeze(v, 2)
        v2 = v2.expand_as(v1)

        q = torch.unsqueeze(q, 1)
        q = torch.unsqueeze(q, 1)
        q = q.expand(batch_size, num_objects, num_objects, q.size(3))

        pairs = torch.cat([v1, v2, q], dim=3)
        pairs = pairs.view(batch_size * num_objects * num_objects, -1)
        out = self.g(pairs)
        out = out.view(batch_size, num_objects, num_objects, -1)

        return out


if __name__ == '__main__':
    from torch.autograd import Variable

    v = torch.rand(10, 3, 1024)
    q = torch.rand(10, 512)
    rn = RelationalNet(1024, 512, 2)
    out = rn.forward(Variable(v), Variable(q))
    print out.size()



