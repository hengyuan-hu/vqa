import torch
import torch.nn as nn
from torch.autograd import Variable
from top_down_attention import TopDownAttention
from attention import NewAttention2, UniAttention
from language_model import WordEmbedding, QuestionEmbedding
# from glu import GLU
from classifier import SimpleClassifier
from relation import RelationModule
from fc import FCNet


class RelationModel0(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, relation, q_net, v_net, classifier):
        super(RelationModel0, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier

        self.relation = relation

    def forward(self, v, b, det, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)
        q_repr = self.q_net(q_emb)

        v_att = self.v_att(v, q_emb).unsqueeze(2) # [batch, k, 1]
        b = b[:, :, :4] # area is not used
        relation_matrix = self.relation(b, q_emb) # [batch, k, k]
        prop_att = torch.bmm(relation_matrix, v_att) # [batch, k, 1]

        # v_emb = (sementic_att * v).sum(1) # [batch, v_dim]
        # v_repr = self.v_net(v_emb)

        rv_emb = (prop_att * v).sum(1)
        rv_repr = self.v_net(rv_emb)
        joint_repr = q_repr * rv_repr

        # joint_repr = q_repr * (v_repr + rv_repr)
        logits = self.classifier(joint_repr)
        return logits



def build_rm0(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)

    v_att = NewAttention2(dataset.v_dim, q_emb.num_hid, num_hid)
    relation = RelationModule(None, q_emb.num_hid, None)
    q_net = FCNet([q_emb.num_hid, num_hid], 0)
    v_net = FCNet([dataset.v_dim, num_hid], 0)
    classifier = SimpleClassifier(num_hid, num_hid * 2, dataset.num_ans_candidates)

    return RelationModel0(w_emb, q_emb, v_att, relation, q_net, v_net, classifier)
