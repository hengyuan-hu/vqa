import torch
import torch.nn as nn
from top_down_attention import TopDownAttention
from language_model import QuestionEmbedding
from glu import GLU
from classifier import SimpleClassifier
from ram import RAM


class BaseModel(nn.Module):
    def __init__(self, q_emb, v_att, q_net, v_net, classifier):
        super(BaseModel, self).__init__()
        self.q_emb = q_emb
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
        q_repr = self.q_net(q_emb)

        v_emb = self.v_att(v, q_emb).sum(1) # [batch, v_dim]
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        return joint_repr


class GateInputModel(nn.Module):
    def __init__(self, q_emb, v_gate, v_att, q_net, v_net, classifier):
        super(GateInputModel, self).__init__()
        self.q_emb = q_emb
        self.v_gate = v_gate
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
        q_repr = self.q_net(q_emb)

        batch, k, v_dim = v.size()
        q_expand = q_emb.unsqueeze(1).repeat(1, k, 1)
        gate_input = torch.cat([q_expand, v], 2)
        gate_input = gate_input.view(batch*k, -1)
        v_gate = self.v_gate(gate_input)
        v_gate = v_gate.view(batch, k, -1)
        # print v_gate.size()
        # print v_gate[0][0][:10]
        v = v * v_gate

        v_emb = self.v_att(v, q_emb).sum(1) # [batch, v_dim]
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        return joint_repr


def build_baseline0(dataset, num_hid):
    q_emb = QuestionEmbedding(dataset.dictionary.ntoken, 300, num_hid, 1, False)
    v_att = TopDownAttention(q_emb.nhid, dataset.v_dim, num_hid)
    q_net = GLU(q_emb.nhid, num_hid)
    v_net = GLU(dataset.v_dim, num_hid)
    classifier = SimpleClassifier(num_hid, num_hid * 2, dataset.num_ans_candidates)
    return BaseModel(q_emb, v_att, q_net, v_net, classifier)


def build_gateinput0(dataset, num_hid):
    q_emb = QuestionEmbedding(dataset.dictionary.ntoken, 300, num_hid, 1, False)
    v_att = TopDownAttention(q_emb.nhid, dataset.v_dim, num_hid)
    q_net = GLU(q_emb.nhid, num_hid)
    v_net = GLU(dataset.v_dim, num_hid)
    classifier = SimpleClassifier(num_hid, num_hid * 2, dataset.num_ans_candidates)

    v_gate = nn.Sequential(
        GLU(q_emb.nhid + dataset.v_dim, dataset.v_dim),
        nn.Linear(dataset.v_dim, dataset.v_dim),
        nn.Sigmoid()
    )

    return GateInputModel(q_emb, v_gate, v_att, q_net, v_net, classifier)


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
