import copy
import torch
import torch.nn as nn
from torch.autograd import Variable
from top_down_attention import TopDownAttention
from attention import NewAttention, UniAttention
from language_model import QuestionEmbedding
from glu import GLU
from classifier import SimpleClassifier
from relation import RelationModule


class BaseModel(nn.Module):
    def __init__(self, q_emb,
                 q_sementic_att, q_relation_att,
                 v_att, relation, q_net, v_net, classifier):
        super(BaseModel, self).__init__()
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

        sementic_att = self.v_att.logits(v, q_sementic_emb).unsqueeze(2) #[batch, k, 1]
        relation_att = self.relation(b, q_relation_emb) # [batch, k, k]
        # relation_att = Variable(torch.eye(relation_att.size(1))).cuda()
        # relation_att = relation_att.repeat(sementic_att.size(0), 1, 1)
        prop_att = torch.bmm(relation_att, sementic_att).squeeze(2) # [batch, k]
        prop_att = prop_att + sementic_att.squeeze(2)
        prop_att = nn.functional.softmax(prop_att).unsqueeze(2) # [batch, k, 1]
        v_emb = (prop_att * v).sum(1) # [batch, v_dim]
        v_repr = self.v_net(v_emb)

        joint_repr = q_repr * v_repr
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

    return BaseModel(q_emb, q_att1, q_att2, v_att, relation, q_net, v_net, classifier)
