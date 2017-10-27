import torch
import torch.nn as nn
from top_down_attention import TopDownAttention
from language_model import QuestionEmbedding
from glu import GLU
from classifier import SimpleClassifier
from ram import RAM


class ReAttendModel(nn.Module):
    def __init__(self, q_emb, q_att, v_att, q_net, v_net, classifier):
        super(ReAttendModel, self).__init__()
        self.q_emb = q_emb
        self.q_att = q_att
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier

    def forward(self, v, q):
        """Forward

        v: [batch_size, num_objs, obj_dim]
        q: [sentence_length, batch_size]
        """
        joint_repr = self._forward(v, q)
        pred = self.classifier(joint_repr)
        return pred

    def loss(self, v, q, labels):
        joint_repr = self._forward(v, q)
        loss = self.classifier.loss(joint_repr, labels)
        return loss

    def _forward(self, v, q):
        # [seq_length, batch, q_dim]
        q_annot = self.q_emb.forward_allout(q)

        # get initial q_emb
        q_emb = q_annot[-1]
        # print q_emb.size()
        # get first attended v
        v_emb = self.v_att(v, q_emb).sum(1) # [batch, v_dim]
        # get re-attended q (TODO: stop gradient for v_atted here?)
        # v_atted = torch.autograd.Variable(v_atted.data)
        q_annot = q_annot.transpose(0, 1)
        q_atted = self.q_att(q_annot, v_emb).sum(1)
        # print q_atted.size()
        # get re-attended v
        v_atted = self.v_att(v, q_atted).sum(1)

        q_att_repr = self.q_net(q_atted)
        v_att_repr = self.v_net(v_atted)
        joint_repr = q_att_repr * v_att_repr
        return joint_repr


class ReAttendModel2(ReAttendModel):
    def _forward(self, v, q):
        # [seq_length, batch, q_dim]
        q_annot = self.q_emb.forward_allout(q)

        # get initial q_emb
        q_emb = q_annot[-1]
        # print q_emb.size()
        # get first attended v
        v_emb = self.v_att(v, q_emb).sum(1) # [batch, v_dim]
        # get re-attended q (TODO: stop gradient for v_atted here?)
        # v_atted = torch.autograd.Variable(v_atted.data)
        q_annot = q_annot.transpose(0, 1)
        q_atted = self.q_att(q_annot, v_emb).sum(1)
        # print q_atted.size()
        # get re-attended v
        v_atted = self.v_att(v, q_atted).sum(1)

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr

        q_att_repr = self.q_net(q_atted)
        v_att_repr = self.v_net(v_atted)
        joint_att_repr = q_att_repr * v_att_repr

        return torch.cat([joint_repr, joint_att_repr], 1)


def build_reatt0(dataset, num_hid):
    q_emb = QuestionEmbedding(dataset.dictionary.ntoken, 300, num_hid, 1, False)
    v_att = TopDownAttention(q_emb.nhid, dataset.v_dim, num_hid)
    q_att = TopDownAttention(dataset.v_dim, q_emb.nhid, num_hid)

    q_net = GLU(q_emb.nhid, num_hid)
    v_net = GLU(dataset.v_dim, num_hid)
    classifier = SimpleClassifier(num_hid, num_hid * 2, dataset.num_ans_candidates)
    return ReAttendModel(q_emb, q_att, v_att, q_net, v_net, classifier)


def build_reatt2(dataset, num_hid):
    q_emb = QuestionEmbedding(dataset.dictionary.ntoken, 300, num_hid, 1, False)
    v_att = TopDownAttention(q_emb.nhid, dataset.v_dim, num_hid)
    q_att = TopDownAttention(dataset.v_dim, q_emb.nhid, num_hid)

    q_net = GLU(q_emb.nhid, num_hid)
    v_net = GLU(dataset.v_dim, num_hid)
    classifier = SimpleClassifier(num_hid*2, num_hid * 2, dataset.num_ans_candidates)
    return ReAttendModel2(q_emb, q_att, v_att, q_net, v_net, classifier)
