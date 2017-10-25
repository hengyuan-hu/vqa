import torch
import torch.nn as nn
from top_down_attention import TopDownAttention
from language_model import QuestionEmbedding
from glu import GLU
from classifier import SimpleClassifier


class BaseModel(nn.Module):
    def __init__(self, q_emb, v_attention, q_net, v_net, classifier):
        super(BaseModel, self).__init__()
        self.q_emb = q_emb
        self.v_attention = v_attention
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
        q_emb = self.q_emb(q) # [batch, q_dim]
        q_repr = self.q_net(q_emb)

        v_emb = self.v_attention(v, q_emb).sum(1) # [batch, v_dim]
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        return joint_repr


def build_baseline0(dataset, num_hid):
    q_emb = QuestionEmbedding(dataset.dictionary.ntoken, 300, num_hid, 1, False)
    v_attention = TopDownAttention(q_emb.nhid, dataset.v_dim, num_hid)
    q_net = GLU(q_emb.nhid, num_hid)
    v_net = GLU(dataset.v_dim, num_hid)
    classifier = SimpleClassifier(num_hid, num_hid * 2, dataset.num_ans_candidates)
    return BaseModel(q_emb, v_attention, q_net, v_net, classifier)


# Not very useful
def build_baseline0_bidirect(dataset, num_hid):
    q_emb = QuestionEmbedding(dataset.dictionary.ntoken, 300, num_hid, 1, True)
    v_attention = TopDownAttention(q_emb.nhid * 2, dataset.v_dim, num_hid)
    q_net = GLU(q_emb.nhid * 2, num_hid)
    v_net = GLU(dataset.v_dim, num_hid)
    classifier = SimpleClassifier(num_hid, num_hid * 2, dataset.num_ans_candidates)
    return BaseModel(q_emb, v_attention, q_net, v_net, classifier)
