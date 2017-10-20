import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import utils


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout, tie_weights):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        assert rnn_type in ['LSTM', 'GRU']
        self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        if tie_weights:
            assert nhid == ninp, \
                'When using the tied flag, nhid must be equal to emsize'
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        print emb.size()
        output, hidden = self.rnn(emb, hidden)
        print output.size()
        output = self.drop(output)
        decoded = self.decoder(
            output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())


class QuestionEmbedding(nn.Module):
    def __init__(self, ntoken, emb_dim, nhid, nlayers, rnn_type='GRU'):
        """Module for question embedding

        The ntoken-th dim is used for padding_idx, which agrees *implicitly*
        with the definition in Dictionary.
        """
        super(QuestionEmbedding, self).__init__()
        self.emb = nn.Embedding(ntoken+1, emb_dim, padding_idx=ntoken)
        self.rnn = nn.GRU(emb_dim, nhid, nlayers) # TODO: dropout?

        self.ntoken = ntoken
        self.emb_dim = emb_dim
        self.nhid = nhid
        self.nlayers = nlayers
        self.rnn_type = rnn_type

    def init_embedding(self, np_file):
        weight_init = torch.from_numpy(np.load(np_file))
        utils.assert_eq(weight_init.shape, (self.ntoken, self.emb_dim))
        self.emb.weight.data[:self.ntoken] = weight_init

    def init_hidden(self, batch):
        # just to get the type of tensor
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, batch, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, batch, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, batch, self.nhid).zero_())

    def forward(self, x):
        batch = x.size(1) # x: [sequence_length, batch]
        hidden = self.init_hidden(batch)
        emb = self.emb(x)
        # emb: [sequence, batch, emb_dim]
        output, hidden = self.rnn(emb, hidden)
        return output[-1]


if __name__ == '__main__':
    from dataset import Dictionary, VQAFeatureDataset

    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    dset = VQAFeatureDataset('dev', dictionary)

    batch = 20
    dataloader = torch.utils.data.DataLoader(
        dset, batch_size=batch, shuffle=True, num_workers=4, drop_last=False)

    i, (v, q, a) = next(enumerate(dataloader))
    model = QuestionEmbedding(dictionary.ntoken, 300, 512, 1)
    model.init_embedding('data/glove6b_init_300d.npy')

    print q.size()
    q = q.t() # [sequence, batch]
    y = model.forward(Variable(q))
    print y.size()
