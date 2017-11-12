import os
import sys
import argparse
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.weight_norm import weight_norm
from torch.utils.data import DataLoader
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import Logger


class Classifier(nn.Module):
    def __init__(self, in_dim, num_hid, num_layers, num_cls):
        super(Classifier, self).__init__()

        layers = [
            weight_norm(nn.Linear(in_dim, num_hid), dim=None),
            nn.ReLU()]
        for i in range(1, num_layers):
            layers.append(weight_norm(nn.Linear(num_hid, num_hid), dim=None))
            layers.append(nn.ReLU())
        layers.append(weight_norm(nn.Linear(num_hid, num_cls)))

        self.main = nn.Sequential(*layers)
        self.num_cls = num_cls

    def forward(self, x):
        logits = self.main(x)
        return logits


def compute_score_with_logits(logits, y):
    """
    logits: [batch*k, num_cls]
    y: [batch*k]
    """

    _, pred = logits.max(1)
    pred = pred.data.cpu().numpy()
    y = y.data.cpu().numpy()
    assert pred.shape == y.shape
    acc = (y == pred).astype(np.float32)
    return acc


def train(model, train_dset, eval_dset, num_epochs, batch_size, logger, output):
    optim = torch.optim.Adamax(model.parameters())

    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=1)
    eval_loader = DataLoader(eval_dset, batch_size, shuffle=True, num_workers=1)

    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0
        t = time.time()

        for x, y in iter(train_loader):
            x = Variable(x).cuda()
            y = Variable(y).cuda().view(-1)
            logits = model.forward(x).view(-1, model.num_cls)

            loss = nn.functional.cross_entropy(logits, y)
            loss.backward()
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(logits, y).sum()
            total_loss += loss.data[0] * x.size(0)
            train_score += batch_score

        total_loss /= len(train_dset)
        train_score /= len(train_dset)
        eval_score = evaluate(model, eval_loader)

        print 'epoch:', epoch
        print 'loss:', total_loss
        print 'train score:', train_score
        print 'eval score:', eval_score


def evaluate(model, dataloader):
    model.train(False)

    score = 0
    num_img = 0
    for x, y in iter(dataloader):
        x = Variable(x).cuda()
        y = Variable(y).cuda().view(-1)
        logits = model.forward(x).view(-1, model.num_cls)
        batch_score = compute_score_with_logits(logits, y).sum()
        score += batch_score
        num_img += x.size(0)

    model.train(True)
    score = score / num_img
    return score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--log', type=str, default='logs/')
    parser.add_argument('--num_hid', type=int, default=512)
    parser.add_argument('--num_layer', type=int, default=2)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')

    args = parser.parse_args()
    args.log += 'hid%d_layer%d.txt'

    return args


if __name__ == '__main__':
    from dataset import VQAImageDataset

    args = parse_args()

    model = Classifier(2048, args.num_hid, args.num_layer, 81).cuda()
    train_dset = VQAImageDataset('train')
    eval_dset = VQAImageDataset('val')

    logger = Logger(args.log)
    train(model, train_dset, eval_dset, 10, 256, logger, None)
