import time
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
from dataset import VQAFilteredDataset
import gc


def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def train(model, train_dset, eval_dset, num_epochs, batch_size, logger, save_path=None):
    # optim = torch.optim.Adadelta(model.parameters())
    # optim = torch.optim.Adam(model.parameters())
    optim = torch.optim.Adamax(model.parameters())

    spatial_dset = VQAFilteredDataset(eval_dset, utils.spatial_filter)
    action_dset = VQAFilteredDataset(eval_dset, utils.action_filter)

    train_loader = torch.utils.data.DataLoader(
        train_dset, batch_size, shuffle=True, num_workers=1)
    eval_loader =  torch.utils.data.DataLoader(
        eval_dset, batch_size, shuffle=True, num_workers=1)
    spatial_loader =  torch.utils.data.DataLoader(
        spatial_dset, batch_size, shuffle=True, num_workers=1)
    action_loader =  torch.utils.data.DataLoader(
        action_dset, batch_size, shuffle=True, num_workers=1)

    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0
        t = time.time()

        for i, (v, b, q, a) in enumerate(train_loader):
            v = Variable(v).cuda()
            b = Variable(b).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()
            pred = model(v, b, q, a)
            loss = instance_bce_with_logits(pred, a)

            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(pred, a.data).sum()
            total_loss += loss.data[0] * v.size(0)
            train_score += batch_score

        gc.collect()
        total_loss /= len(train_dset)
        train_score /= len(train_dset)
        eval_score, eval_bound = evaluate(model, eval_loader)
        spatial_score, spatial_bound = evaluate(model, spatial_loader)
        action_score, action_bound = evaluate(model, action_loader)

        print logger.log('epoch %d, time: %.2f' % (epoch, time.time()-t))
        print logger.log(
            'train_loss: %.2f, train_score: %.2f, eval_score: %.2f (%.2f)'
            % (total_loss, 100*train_score, 100*eval_score, 100*eval_bound)
        )
        print logger.log(
            'spatial_score: %.2f (%.2f), action_score: %.2f (%.2f)' %
            (100*spatial_score, 100*spatial_bound, 100*action_score, 100*action_bound)
        )

        if save_path is not None:
            print 'saving model...'
            torch.save(model.state_dict(), save_path + '_epoch' + str(epoch) + '.pt')


def evaluate(model, dataloader):
    model.train(False)

    score = 0
    upper_bound = 0
    num_data = 0
    for v, b, q, a in iter(dataloader):
        v = Variable(v, volatile=True).cuda()
        b = Variable(b, volatile=True).cuda()
        q = Variable(q, volatile=True).cuda()
        pred = model(v, b, q, None)
        batch_score = compute_score_with_logits(pred, a.cuda()).sum()
        score += batch_score
        upper_bound += (a.max(1)[0]).sum()
        num_data += pred.size(0)

    model.train(True)

    score = score / num_data
    upper_bound = upper_bound / num_data
    return score, upper_bound
