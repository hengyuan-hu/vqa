import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class QuestionEmbedding(nn.Module):
    def __init__(self, ntoken, emb_dim, nhid, nlayers, bidirect, rnn_type='GRU'):
        """Module for question embedding

        The ntoken-th dim is used for padding_idx, which agrees *implicitly*
        with the definition in Dictionary.
        """
        super(QuestionEmbedding, self).__init__()
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU

        self.emb = nn.Embedding(ntoken+1, emb_dim, padding_idx=ntoken)
        self.rnn = rnn_cls(
            emb_dim, nhid, nlayers, bidirectional=bidirect, batch_first=True)

        self.ntoken = ntoken
        self.emb_dim = emb_dim
        self.nhid = nhid
        self.nlayers = nlayers
        self.rnn_type = rnn_type
        self.ndirections = 1 + int(bidirect)

    def init_embedding(self, np_file):
        weight_init = torch.from_numpy(np.load(np_file))
        assert weight_init.shape == (self.ntoken, self.emb_dim)
        self.emb.weight.data[:self.ntoken] = weight_init

    def init_hidden(self, batch):
        # just to get the type of tensor
        weight = next(self.parameters()).data
        hid_shape = (self.nlayers * self.ndirections, batch, self.nhid)
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(*hid_shape).zero_()),
                    Variable(weight.new(*hid_shape).zero_()))
        else:
            return Variable(weight.new(*hid_shape).zero_())

    def forward(self, x):
        # x: [batch, sequence_length]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        # print x.size()
        emb = self.emb(x)
        # emb: [batch, sequence, emb_dim]
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(emb, hidden)
        if self.ndirections == 1:
            return output[:, -1]

        forward_ = output[:, -1, :self.nhid]
        backward = output[:, 0, self.nhid:]
        emb = torch.cat((forward_, backward), dim=1)
        return emb

    def forward_allout(self, x):
        assert self.ndirections == 1, 'bidirection not supported yet'
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        emb = self.emb(x)
        # emb: [sequence, batch, emb_dim]
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(emb, hidden)
        return output



if __name__ == '__main__':
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dataset import Dictionary, VQAFeatureDataset

    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    dset = VQAFeatureDataset('dev', dictionary)

    batch = 20
    dataloader = torch.utils.data.DataLoader(
        dset, batch_size=batch, shuffle=True, num_workers=4, drop_last=False)

    i, (v, q, a) = next(enumerate(dataloader))
    model = QuestionEmbedding(dictionary.ntoken, 300, 512, 1, True)
    model.init_embedding('data/glove6b_init_300d.npy')

    print q.size()
    q = q.t() # [sequence, batch]
    y = model.forward(Variable(q))
    print y.size()
