import time
import torch
from torch.autograd import Variable


def train(model, train_dset, eval_dset, num_epochs, batch_size, logger):
    # optim = torch.optim.Adadelta(model.parameters())
    # optim = torch.optim.Adam(model.parameters())
    optim = torch.optim.Adamax(model.parameters())
    train_loader = torch.utils.data.DataLoader(
        train_dset, batch_size, shuffle=True, num_workers=1)
    eval_loader =  torch.utils.data.DataLoader(
        eval_dset, batch_size, shuffle=True, num_workers=1)

    for epoch in range(num_epochs):
        total_loss = 0
        t = time.time()
        for i, (v, b, q, a) in enumerate(train_loader):
            v = Variable(v).cuda()
            b = Variable(b).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()
            loss = model(v, b, q, a).mean()
            total_loss += loss.data[0] * v.size(0)

            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

        total_loss /= len(train_dset)
        train_score, upper_bound = evaluate(model, train_loader)
        eval_score, _ = evaluate(model, eval_loader)

        print logger.log('epoch %d, time: %.2f' % (epoch, time.time()-t))
        print logger.log('train_loss: %.2f, train_score: %.2f, eval_score: %.2f (%.2f)'
                         % (total_loss, train_score, eval_score, upper_bound))


def evaluate(model, dataloader):
    model.train(False)

    score = 0
    upper_bound = 0
    for v, b, q, a in iter(dataloader):
        v = Variable(v, volatile=True).cuda()
        b = Variable(b, volatile=True).cuda()
        q = Variable(q, volatile=True).cuda()
        pred = model(v, b, q, None)
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
