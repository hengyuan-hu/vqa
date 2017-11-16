import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm


class FCNet(nn.Module):
    def __init__(self, dims, dropout):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims)-2):
            in_dim = dims[i]
            out_dim = dims[i+1]
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            layers.append(nn.ReLU())
        if len(dims) > 2 and dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


if __name__ == '__main__':
    fc1 = FCNet([10, 20, 10], 0.5)
    print fc1

    print '============'
    fc2 = FCNet([10, 20], 0.5)
    print fc2
