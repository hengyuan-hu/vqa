import copy
import torch
import torch.nn as nn
from torch.autograd import Variable
from top_down_attention import TopDownAttention
from attention import NewAttention, UniAttention
from language_model import WordEmbedding, QuestionEmbedding
from glu import GLU
from classifier import SimpleClassifier
from relation import RelationModule


class RelationModel0(nn.Module):
    def __init__(self, q_emb,
                 q_sementic_att, q_relation_att,
                 v_att, relation, q_net, v_net, classifier):
        super(RelationModel0, self).__init__()
        self.q_emb = q_emb
        self.q_sementic_att = q_sementic_att
        self.q_relation_att = q_relation_att

        self.v_att = v_att
        self.relation = relation

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
        w_emb, q_annotations = self.q_emb.forward_allout(q) # [batch, sequence, q_dim]
        # [batch, sequence, 1]
        q_sementic_att = self.q_sementic_att(q_annotations).unsqueeze(2)
        q_relation_att = self.q_relation_att(q_annotations).unsqueeze(2)
        q_sementic_emb = (w_emb * q_sementic_att).sum(1)
        q_relation_emb = (w_emb * q_relation_att).sum(1)
        q_joint_emb = torch.cat([q_sementic_emb, q_relation_emb], 1)
        q_repr = self.q_net(q_joint_emb)

        sementic_att = self.v_att(v, q_sementic_emb).unsqueeze(2) #[batch, k, 1]
        relation_att = self.relation(b, q_relation_emb) # [batch, k, k]
        prop_att = torch.bmm(relation_att, sementic_att) # [batch, k, 1]
        v_emb = (sementic_att * v).sum(1) # [batch, v_dim]
        v_repr = self.v_net(v_emb)

        rv_emb = (prop_att * v).sum(1)
        rv_repr = self.v_net(rv_emb)

        joint_repr = q_repr * (v_repr + rv_repr)
        logits = self.classifier(joint_repr)
        return logits


class RelationModel1(nn.Module):
    def __init__(self, w_emb, q_sementic_emb, q_relation_emb,
                 v_att, relation, q_net, v_net, classifier):
        super(RelationModel1, self).__init__()
        self.w_emb = w_emb
        self.q_sementic_emb = q_sementic_emb
        self.q_relation_emb = q_relation_emb

        self.v_att = v_att
        self.relation = relation

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
        w_emb = self.w_emb(q)
        q_sementic_emb = self.q_sementic_emb(w_emb) # [batch, q_dim]
        q_relation_emb = self.q_relation_emb(w_emb) # [batch, q_dim]

        # q_joint_emb = torch.cat([q_sementic_emb, q_relation_emb], 1)
        q_sementic_repr = self.q_net(q_sementic_emb)
        q_relation_repr = self.q_net(q_relation_emb)

        sementic_att = self.v_att(v, q_sementic_emb).unsqueeze(2) #[batch, k, 1]
        relation_att = self.relation(b, q_relation_emb) # [batch, k, k]
        prop_att = torch.bmm(relation_att, sementic_att) # [batch, k, 1]
        v_emb = (sementic_att * v).sum(1) # [batch, v_dim]
        v_repr = self.v_net(v_emb)

        rv_emb = (prop_att * v).sum(1)
        rv_repr = self.v_net(rv_emb)

        joint_repr = q_sementic_repr * v_repr + q_relation_repr * rv_repr
        logits = self.classifier(joint_repr)
        return logits


def build_rm0(dataset, num_hid):
    q_emb = QuestionEmbedding(dataset.dictionary.ntoken, 300, num_hid, 1, False)
    q_att1 = UniAttention(q_emb.nhid)
    q_att2 = UniAttention(q_emb.nhid)

    v_att = NewAttention(dataset.v_dim, 300)
    relation = RelationModule(dataset.s_dim, 300, 128)
    q_net = GLU(600, num_hid)
    v_net = GLU(dataset.v_dim, num_hid)
    classifier = SimpleClassifier(num_hid, num_hid * 2, dataset.num_ans_candidates)

    return RelationModel0(
        q_emb, q_att1, q_att2, v_att, relation, q_net, v_net, classifier)


def build_rm1(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb1 = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    q_emb2 = QuestionEmbedding(300, num_hid, 1, False, 0.0)

    v_att = NewAttention(dataset.v_dim, num_hid)
    relation = RelationModule(dataset.s_dim, num_hid, 128)
    q_net = GLU(num_hid, num_hid)
    v_net = GLU(dataset.v_dim, num_hid)
    classifier = SimpleClassifier(num_hid, num_hid * 2, dataset.num_ans_candidates)

    return RelationModel1(
        w_emb, q_emb1, q_emb2, v_att, relation, q_net, v_net, classifier)
