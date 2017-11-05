import torch
import torch.nn as nn
from top_down_attention import TopDownAttention
from attention import NewAttention
from language_model import QuestionEmbedding
from glu import GLU
from classifier import SimpleClassifier
from relation import RelationModule


class BaseModel(nn.Module):
    def __init__(self, q_emb, v_att, relation, q_net, v_net, classifier):
        super(BaseModel, self).__init__()
        self.q_emb = q_emb
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
        q_emb = self.q_emb(q) # [batch, q_dim]
        q_repr = self.q_net(q_emb)

        att_logits = self.v_att.logits(v, q_emb).unsqueeze(2) # [batch, k, 1]
        relation_mat = self.relation(b, q_emb) # [batch, k, k]
        prop_logits = torch.bmm(relation_mat, att_logits).squeeze(2) # [batch, k]
        prop_att = nn.functional.softmax(prop_logits).unsqueeze(2) # [batch, k, 1]

        v_emb = (prop_att * v).sum(1) # [batch, v_dim]
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits


def build_rm0(dataset, num_hid):
    q_emb = QuestionEmbedding(dataset.dictionary.ntoken, 300, num_hid, 1, False)
    v_att = NewAttention(dataset.v_dim, q_emb.nhid)
    relation = RelationModule(dataset.s_dim, q_emb.nhid, 128)
    q_net = GLU(q_emb.nhid, num_hid)
    v_net = GLU(dataset.v_dim, num_hid)
    classifier = SimpleClassifier(num_hid, num_hid * 2, dataset.num_ans_candidates)
    return BaseModel(q_emb, v_att, relation, q_net, v_net, classifier)
