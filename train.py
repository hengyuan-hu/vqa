import time
import torch
from torch.autograd import Variable


def train(model, train_dset, eval_dset, num_epochs, batch_size, eval_batch_size, logger):
    # optim = torch.optim.Adadelta(model.parameters())
    # optim = torch.optim.Adam(model.parameters())
    optim = torch.optim.Adamax(model.parameters())

    for epoch in range(num_epochs):
        dataloader = torch.utils.data.DataLoader(
            train_dset, batch_size, shuffle=True, num_workers=4)

        total_loss = 0
        t = time.time()
        for i, (v, q, a) in enumerate(dataloader):
            v = Variable(v.cuda())
            q = Variable(q.t().cuda())
            a = Variable(a.cuda())
            loss = model.loss(v, q, a)

            loss.backward()
            optim.step()
            optim.zero_grad()

            total_loss += loss.data[0] * v.size(0)

        total_loss /= len(train_dset)
        score, upper_bound = evaluate(model, eval_dset, eval_batch_size)

        print logger.log('epoch %d, time: %.2f' % (epoch, time.time()-t))
        print logger.log('train_loss: %.2f, eval_score: %.2f (%.2f)'
                         % (total_loss, score, upper_bound))


def evaluate(model, eval_dset, eval_batch_size):
    model.train(False)

    dataloader = torch.utils.data.DataLoader(eval_dset, eval_batch_size, num_workers=4)
    score = 0
    upper_bound = 0
    for i, (v, q, a) in enumerate(dataloader):
        v = Variable(v.cuda(), volatile=True)
        q = Variable(q.t().cuda(), volatile=True)
        pred = model(v, q)
        pred = torch.max(pred, 1)[1].data # argmax
        one_hot = torch.zeros(*a.size()).cuda()
        one_hot.scatter_(1, pred.view(-1, 1), 1)

        batch_score = (one_hot * a.cuda()).sum()
        score += batch_score
        upper_bound += (a.max(1)[0]).sum()

    model.train(True)

    score = 100.0 * score / len(eval_dset)
    upper_bound = 100.0 * upper_bound / len(eval_dset)
    return score, upper_bound
