import torch
import torch.nn as nn
from top_down_attention import TopDownAttention
from language_model import QuestionEmbedding
from glu import GLU
from classifier import SimpleClassifier
from ram import RAM


class RAMModel(nn.Module):
    def __init__(self, q_emb, v_attention, v_net, classifier):
        super(RAMModel, self).__init__()
        self.q_emb = q_emb
        self.v_attention = v_attention
        # self.q_net = q_net
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
        q_emb = self.q_emb(q) # [batch, q_dim]
        # q_repr = self.q_net(q_emb)

        v_emb = self.v_attention(v, q_emb).sum(1) # [batch, v_dim]
        v_repr = self.v_net(v_emb)
        return v_repr
        # joint_repr = q_repr * v_repr
        # return joint_repr


def build_ram0(dataset, num_hid):
    q_emb = QuestionEmbedding(dataset.dictionary.ntoken, 300, num_hid, 1, False)

    rnet_in_dim = dataset.v_dim * 2 + num_hid
    r_net = nn.Sequential(
        GLU(rnet_in_dim, dataset.v_dim),
        # GLU(dataset.v_dim, dataset.v_dim)
    )
    v_attention = TopDownAttention(q_emb.nhid, dataset.v_dim, num_hid)
    ram = RAM(v_attention, r_net)

    v_net = GLU(dataset.v_dim, num_hid)
    classifier = SimpleClassifier(num_hid, num_hid * 2, dataset.num_ans_candidates)
    return RAMModel(q_emb, ram, v_net, classifier)
