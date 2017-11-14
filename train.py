import os
import time
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable


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


def train(model, train_loader, eval_loaders, num_epochs, output):
    # optim = torch.optim.Adam(model.parameters())
    optim = torch.optim.Adamax(model.parameters())
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0

    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0
        t = time.time()

        for i, (v, b, det, q, a) in enumerate(train_loader):
            v = Variable(v).cuda()
            b = Variable(b).cuda()
            det = Variable(det).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()

            pred = model(v, b, det, q, a)

            loss = instance_bce_with_logits(pred, a)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(pred, a.data).sum()
            total_loss += loss.data[0] * v.size(0)
            train_score += batch_score

        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)
        logger.append('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))

        for eval_name in eval_loaders:
            loader = eval_loaders[eval_name]
            score, bound = evaluate(model, loader)
            logger.append(
                '\t%s_score: %.2f (%.2f)' % (eval_name, 100 * score, 100 * bound))
            if eval_name == 'eval' and score > best_eval_score:
                model_path = os.path.join(output, 'model.pth')
                torch.save(model.state_dict(), model_path)

        print logger.log('epoch %d, time: %.2f' % (epoch, time.time()-t))


def evaluate(model, dataloader):
    model.train(False)

    score = 0
    upper_bound = 0
    num_data = 0
    for v, b, det, q, a in iter(dataloader):
        v = Variable(v, volatile=True).cuda()
        b = Variable(b, volatile=True).cuda()
        det = Variable(det, volatile=True).cuda()
        q = Variable(q, volatile=True).cuda()
        pred = model(v, b, det, q, None)
        batch_score = compute_score_with_logits(pred, a.cuda()).sum()
        score += batch_score
        upper_bound += (a.max(1)[0]).sum()
        num_data += pred.size(0)

    model.train(True)

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    return score, upper_bound
