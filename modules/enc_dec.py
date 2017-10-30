import torch
import torch.nn as nn
from torch.autograd import Variable
from attention import CrossAttention
from language_model import QuestionEmbedding
from glu import GLU
from classifier import SimpleClassifier


class EncFcDec(nn.Module):
    def __init__(self, q_emb, v_att, r_net, classifier):
        super(EncFcDec, self).__init__()

        self.q_emb = q_emb
        self.v_att = v_att
        self.r_net = r_net
        self.classifier = classifier

    def forward(self, v, q, labels):
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
        q_emb = self.q_emb.forward_allout(q)
        v_att = self.v_att(v, q_emb)

        # print q_emb.size()
        # print v_att.size()
        batch, l, q_dim = q_emb.size()
        vq = torch.cat([q_emb, v_att], 2)
        vq = vq.view(batch*l, -1)
        vq = self.r_net(vq).view(batch, l, -1).sum(1)
        # print vq.size()
        return vq


def build_efcd(dataset, num_hid):
    q_emb = QuestionEmbedding(dataset.dictionary.ntoken, 300, num_hid, 1, False)
    # q_net = GLU(q_emb.nhid, num_hid)

    rnet_in_dim = dataset.v_dim + num_hid
    r_net = nn.Sequential(
        GLU(rnet_in_dim, num_hid),
        GLU(num_hid, num_hid)
    )
    v_att = CrossAttention(q_emb.nhid, dataset.v_dim, num_hid, 1)

    # v_net = GLU(dataset.v_dim, num_hid)
    classifier = SimpleClassifier(num_hid, num_hid * 2, dataset.num_ans_candidates)
    return EncFcDec(q_emb, v_att, r_net, classifier)
