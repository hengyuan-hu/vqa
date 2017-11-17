import torch
import torch.nn as nn
from top_down_attention import TopDownAttention
from attention import *
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from torch.nn.utils.weight_norm import weight_norm
from fc import FCNet


class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier

        self.w_att = UniAttention(self.q_emb.num_hid * self.q_emb.ndirections)

    def forward(self, v, b, det, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb.forward_all(w_emb) # [batch, seq_length, q_dim]
        w_att = self.w_att(q_emb).unsqueeze(2)
        w_emb = (w_emb * w_att).sum(1)
        q_repr = self.q_net(q_emb[:, -1])

        # v = torch.cat([v, b], 2)
        att = self.v_att(v, w_emb).unsqueeze(2).expand_as(v)
        v_emb = (att * v).sum(1) # [batch, v_dim]
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits


def build_wemb_newatt2(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention2(dataset.v_dim, 300, num_hid)
    q_net = FCNet([num_hid, num_hid], 0)
    v_net = FCNet([dataset.v_dim, num_hid], 0)
    classifier = SimpleClassifier(num_hid, num_hid * 2, dataset.num_ans_candidates)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)
