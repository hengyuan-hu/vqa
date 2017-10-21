import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import utils


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
