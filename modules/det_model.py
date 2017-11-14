import torch
import torch.nn as nn
from torch.autograd import Variable
from top_down_attention import TopDownAttention
from attention import NewAttention, UniAttention
from language_model import WordEmbedding, QuestionEmbedding
from glu import GLU
from classifier import SimpleClassifier
from relation import RelationModule
from torch.nn.utils.weight_norm import weight_norm


class RelationNet(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, num_layers):
        super(RelationNet, self).__init__()

        self.v_dim = v_dim
        self.q_dim = q_dim
        self.num_hid = num_hid
        in_dim = 2 * v_dim + q_dim
        layers = []
        for _ in range(num_layers):
            # layers.append(weight_norm(nn.Linear(in_dim, num_hid), dim=None))
            # layers.append(nn.ReLU())
            layers.append(GLU(in_dim, num_hid))
            in_dim = num_hid

        self.main = nn.Sequential(*layers)

    def forward(self, v, q):
        """
        v: [batch, k, v_dim]
        q: [batch, q_dim]
        """
        batch, k, _ = v.size()
        vi = v.unsqueeze(1).repeat(1, k, 1, 1)
        vj = v.unsqueeze(2).repeat(1, 1, k, 1)
        q = q.unsqueeze(1).unsqueeze(1).repeat(1, k, k, 1)
        vq_pairs = torch.cat([vi, vj, q], 3) # [batch, k, k, 2v+q]

        relations = self.main(vq_pairs) # [batch, k, k, num_hid]
        relations = relations.view(batch, k*k, self.num_hid).sum(1)

        return relations # [batch, num_hid]


class DetModel1(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, relation, classifier):
        super(DetModel1, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net

        self.relation = relation

        self.classifier = classifier

    def forward(self, v, b, det, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        det: [batch, num_objs]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        # normal baseline model
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]
        q_repr = self.q_net(q_emb) # [batch, num_hid]

        att = self.v_att(v, q_emb).unsqueeze(2) #[batch, k, 1]
        v_emb = (att * v).sum(1) # [batch, v_dim]
        v_repr = self.v_net(v_emb) # [batch, num_hid]

        joint_repr = q_repr + v_repr

        # relational part
        det_w_emb = self.w_emb(det) # [batch, num_objs, w_dim]
        relation_v = torch.cat([det_w_emb, b], 2)
        relation_repr = self.relation(relation_v, q_emb)

        joint_repr = joint_repr + relation_repr

        logits = self.classifier(joint_repr)
        return logits


def build_det1(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, num_hid)
    q_net = GLU(num_hid, num_hid)
    v_net = GLU(dataset.v_dim, num_hid)
    relation = RelationNet(dataset.s_dim + 300, num_hid, num_hid, 2)
    classifier = SimpleClassifier(num_hid, num_hid * 2, dataset.num_ans_candidates)

    return DetModel1(w_emb, q_emb, v_att, q_net, v_net, relation, classifier)