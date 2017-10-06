import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

import dataset
from language_model import RNNModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help='dataset to use, (PTB, VQA)')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (LSTM, GRU)')
    parser.add_argument('--emsize', type=int, default=200,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=200,
                        help='# hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=20)
    parser.add_argument('--clip', type=float, default=0.25, help='max gradient norm')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--bptt', type=int, default=35, help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--tied', action='store_true',
                        help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--save', type=str,  default='model.pt',
                        help='path to save the final model')
    args = parser.parse_args()
    return args


def batchify(data, bsz):
    nbatch = data.shape[0] // bsz
    data = data[:nbatch * bsz]
    data = data.reshape(bsz, -1).transpose()
    return data


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = torch.from_numpy(source[i:i+seq_len]).cuda()
    target = torch.from_numpy(source[i+1:i+1+seq_len].reshape((-1,))).cuda()
    return data, target


def evaluate(data_source, bptt):
    # Turn on evaluation mode which disables dropout.
    model.train(False)
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    for i in range(0, len(data_source) - 1, bptt):
        data, targets = get_batch(data_source, i)
        data = Variable(data, volatile=True)
        targets = Variable(targets)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * nn.functional.cross_entropy(output_flat, targets).data

        hidden = repackage_hidden(hidden)
    model.train()
    return total_loss[0] / len(data_source)


def train(model, corpus, batch_size, lr, bptt, clip_norm):
    train_data = batchify(corpus.train, batch_size)
    total_loss = 0
    # optim = torch.optim.Adam(model.parameters(), lr)
    optim = torch.optim.SGD(model.parameters(), lr)
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)

    for i in range(0, len(train_data) - 1, bptt):
        # print i, len(train_data) / bptt
        data, targets = get_batch(train_data, i)
        output, hidden = model(Variable(data), hidden)
        loss = nn.functional.cross_entropy(output.view(-1, ntokens), Variable(targets))
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), clip_norm)
        optim.step()
        model.zero_grad()
        hidden = repackage_hidden(hidden)
        total_loss += len(data) * loss.data[0]
    return total_loss / len(train_data)


if __name__ == '__main__':
    # Set the random seed manually for reproducibility.
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    if args.dataset == 'VQA':
        dset = dataset.VQADataset()
        corpus = dataset.Corpus(
            dset.get_raw_questions('train'),
            dset.get_raw_questions('valid'))
    elif args.dataset == 'PTB':
        corpus = dataset.PTBCorpus()
    else:
        assert False, 'no such dataset'

    ntokens = len(corpus.dictionary)
    model = RNNModel(
        args.model,
        ntokens,
        args.emsize,
        args.nhid,
        args.nlayers,
        args.dropout,
        args.tied).cuda()

    eval_batch_size = 10
    train_data = batchify(corpus.train, args.batch_size)
    val_data = batchify(corpus.valid, eval_batch_size)

    best_valid_loss = float('inf')
    lr = args.lr
    for epoch in range(1, args.epochs+1):
        t = time.time()
        loss = train(model, corpus, args.batch_size, lr, args.bptt, args.clip)
        print 'epochs: %d, time: %.2f, lr: %s'% (epoch, time.time() - t, lr)
        valid_loss = evaluate(val_data, args.bptt)
        valid_ppl = np.exp(valid_loss)
        print 'train_loss: %.2f, valid_loss: %.2f, valid ppl: %.2f' % (
            loss, valid_loss, valid_ppl)

        if valid_loss < best_valid_loss:
            best_valid_loss= valid_loss
        else:
            lr /= 4.0
