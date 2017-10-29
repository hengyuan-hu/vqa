import torch
import torch.nn as nn
from torch.autograd import Variable
from top_down_attention import TopDownAttention
from language_model import QuestionEmbedding
from glu import GLU
from classifier import SimpleClassifier
# from ram import RAM


class RNNFusion(nn.Module):
    def __init__(self, q_dim, v_dim, nhid, nlayers, bidirect, rnn_type='LSTM'):
        super(RNNFusion, self).__init__()

        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU
        self.rnn = rnn_cls(
            q_dim + v_dim, nhid, nlayers, bidirectional=bidirect, batch_first=True)

        self.nhid = nhid
        self.nlayers = nlayers
        self.rnn_type = rnn_type
        self.ndirections = 1 + int(bidirect)

        self.fc = GLU(self.ndirections * self.nhid, self.nhid)

    def init_hidden(self, batch):
        # just to get the type of tensor
        weight = next(self.parameters()).data
        hid_shape = (self.nlayers * self.ndirections, batch, self.nhid)
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(*hid_shape).zero_()),
                    Variable(weight.new(*hid_shape).zero_()))
        else:
            return Variable(weight.new(*hid_shape).zero_())

    def forward(self, v, q):
        """
        q: [batch, q_dim]
        v: [batch, K, v_dim]

        return: [batch, K, nhid * num_directions]
        """
        batch, k, _ = v.size()
        if q is None:
            x = v
        else:
            q = q.unsqueeze(1).repeat(1, k, 1)
            x = torch.cat([q, v], 2)

        hidden = self.init_hidden(batch)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(x, hidden)

        # output: [batch, k, nhid * num_directions]
        output = output.contiguous()
        output = output.view(batch*k, -1)
        output = self.fc(output).view(batch, k, -1)
        # if self.ndirections == 2:
        #     output = output[:, :, :self.nhid] + output[:, :, self.nhid:]
        return output


class RNNModel0(nn.Module):
    def __init__(self, q_emb, rnn, v_att, q_net, v_net, classifier):
        super(RNNModel0, self).__init__()
        self.q_emb = q_emb
        self.rnn = rnn
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier

    def forward(self, v, q, labels):
        """Forward

        v: [batch_size, num_objs, obj_dim]
        q: [sentence_length, batch_size]
        """
        joint_repr = self._forward(v, q)
        if self.training:
            loss = self.classifier.loss(joint_repr, labels)
            return loss
        else:
            pred = self.classifier(joint_repr)
            return pred

    def loss(self, v, q, labels):
        joint_repr = self._forward(v, q)
        loss = self.classifier.loss(joint_repr, labels)
        return loss

    def _forward(self, v, q):
        q_emb = self.q_emb(q) # [batch, q_dim]
        v_res = self.rnn(v, None)
        v = v + v_res
        # print v.size()
        v_emb = self.v_att(v, q_emb).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        return joint_repr


def build_rnn0(dataset, num_hid):
    q_emb = QuestionEmbedding(dataset.dictionary.ntoken, 300, num_hid, 1, False)
    v_rnn = RNNFusion(0, dataset.v_dim, dataset.v_dim, 1, True, 'GRU')
    v_att = TopDownAttention(q_emb.nhid, dataset.v_dim, num_hid)
    q_net = GLU(q_emb.nhid, num_hid)
    v_net = GLU(dataset.v_dim, num_hid)
    classifier = SimpleClassifier(num_hid, num_hid * 2, dataset.num_ans_candidates)
    return RNNModel0(q_emb, v_rnn, v_att, q_net, v_net, classifier)
