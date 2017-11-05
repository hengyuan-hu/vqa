import torch
import torch.nn as nn
from top_down_attention import TopDownAttention
from attention import NewAttention
from language_model import QuestionEmbedding
from glu import GLU
from classifier import SimpleClassifier


class BaseModel(nn.Module):
    def __init__(self, q_emb, v_att, q_net, v_net, classifier):
        super(BaseModel, self).__init__()
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier

    def forward(self, v, b, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        q_emb = self.q_emb(q) # [batch, q_dim]
        q_repr = self.q_net(q_emb)

        v = torch.cat([v, b], 2)
        att = self.v_att(v, q_emb).unsqueeze(2).expand_as(v)
        v_emb = (att * v).sum(1) # [batch, v_dim]
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits


def build_baseline0(dataset, num_hid):
    q_emb = QuestionEmbedding(dataset.dictionary.ntoken, 300, num_hid, 1, False)
    v_att = TopDownAttention(q_emb.nhid, dataset.v_dim, num_hid)
    q_net = GLU(q_emb.nhid, num_hid)
    v_net = GLU(dataset.v_dim, num_hid)
    classifier = SimpleClassifier(num_hid, num_hid * 2, dataset.num_ans_candidates)
    return BaseModel(q_emb, v_att, q_net, v_net, classifier)


def build_baseline0_dropout(dataset, num_hid):
    q_emb = QuestionEmbedding(dataset.dictionary.ntoken, 300, num_hid, 2, False,
                              dropout=0.5)
    v_att = TopDownAttention(q_emb.nhid, dataset.v_dim, num_hid)
    q_net = GLU(q_emb.nhid, num_hid)
    v_net = GLU(dataset.v_dim, num_hid)
    classifier = SimpleClassifier(num_hid, num_hid * 2, dataset.num_ans_candidates)
    return BaseModel(q_emb, v_att, q_net, v_net, classifier)


def build_baseline0_emb_dropout(dataset, num_hid):
    q_emb = QuestionEmbedding(dataset.dictionary.ntoken, 300, num_hid, 1, False,
                              emb_dropout=0.5)
    v_att = TopDownAttention(q_emb.nhid, dataset.v_dim, num_hid)
    q_net = GLU(q_emb.nhid, num_hid)
    v_net = GLU(dataset.v_dim, num_hid)
    classifier = SimpleClassifier(num_hid, num_hid * 2, dataset.num_ans_candidates)
    return BaseModel(q_emb, v_att, q_net, v_net, classifier)


def build_baseline0_double_dropout(dataset, num_hid):
    q_emb = QuestionEmbedding(dataset.dictionary.ntoken, 300, num_hid, 2, False,
                              emb_dropout=0.2, dropout=0.5)
    v_att = TopDownAttention(q_emb.nhid, dataset.v_dim, num_hid)
    q_net = GLU(q_emb.nhid, num_hid)
    v_net = GLU(dataset.v_dim, num_hid)
    classifier = SimpleClassifier(num_hid, num_hid * 2, dataset.num_ans_candidates)
    return BaseModel(q_emb, v_att, q_net, v_net, classifier)


def build_baseline0_double_dropout2(dataset, num_hid):
    q_emb = QuestionEmbedding(dataset.dictionary.ntoken, 300, num_hid, 2, False,
                              emb_dropout=0.5, dropout=0.5)
    v_att = TopDownAttention(q_emb.nhid, dataset.v_dim, num_hid)
    q_net = GLU(q_emb.nhid, num_hid)
    v_net = GLU(dataset.v_dim, num_hid)
    classifier = SimpleClassifier(num_hid, num_hid * 2, dataset.num_ans_candidates)
    return BaseModel(q_emb, v_att, q_net, v_net, classifier)


def build_baseline0_newatt(dataset, num_hid):
    q_emb = QuestionEmbedding(dataset.dictionary.ntoken, 300, num_hid, 1, False)
    v_att = NewAttention(dataset.v_dim + dataset.s_dim, q_emb.nhid)
    q_net = GLU(q_emb.nhid, num_hid)
    v_net = GLU(dataset.v_dim + dataset.s_dim, num_hid)
    classifier = SimpleClassifier(num_hid, num_hid * 2, dataset.num_ans_candidates)
    return BaseModel(q_emb, v_att, q_net, v_net, classifier)


def build_baseline1(dataset, num_hid):
    q_emb = QuestionEmbedding(dataset.dictionary.ntoken, 300, num_hid, 1, False)
    v_att = TopDownAttention(q_emb.nhid, dataset.v_dim, num_hid)
    q_net = GLU(q_emb.nhid, num_hid)

    v_net = nn.Sequential(
        GLU(dataset.v_dim, dataset.v_dim),
        GLU(dataset.v_dim, num_hid)
    )
    classifier = SimpleClassifier(num_hid, num_hid * 2, dataset.num_ans_candidates)
    return BaseModel(q_emb, v_att, q_net, v_net, classifier)


def build_baseline2(dataset, num_hid):
    """2 layer attention compared with baseline1, no help"""
    q_emb = QuestionEmbedding(dataset.dictionary.ntoken, 300, num_hid, 1, False)
    v_att = TopDownAttention(q_emb.nhid, dataset.v_dim, num_hid, 2)
    q_net = GLU(q_emb.nhid, num_hid)

    v_net = nn.Sequential(
        GLU(dataset.v_dim, dataset.v_dim),
        GLU(dataset.v_dim, num_hid)
    )
    classifier = SimpleClassifier(num_hid, num_hid * 2, dataset.num_ans_candidates)
    return BaseModel(q_emb, v_att, q_net, v_net, classifier)


# Not very useful
def build_baseline0_bidirect(dataset, num_hid):
    q_emb = QuestionEmbedding(dataset.dictionary.ntoken, 300, num_hid, 1, True)
    v_att = TopDownAttention(q_emb.nhid * 2, dataset.v_dim, num_hid)
    q_net = GLU(q_emb.nhid * 2, num_hid)
    v_net = GLU(dataset.v_dim, num_hid)
    classifier = SimpleClassifier(num_hid, num_hid * 2, dataset.num_ans_candidates)
    return BaseModel(q_emb, v_att, q_net, v_net, classifier)
