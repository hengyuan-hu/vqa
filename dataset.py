import os
import json
import torch
import numpy as np


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, train, valid):
        self.dictionary = Dictionary()
        self.train = self.tokenize(train)
        self.valid = self.tokenize(valid)

    def tokenize(self, qset):
        """Tokenizes a question set (list of strings)."""
        tokens = 0
        ids = []
        for q in qset:
            q = q.replace(',', '').replace('?', '').replace('\'s', ' \'s')
            words = q.split() + [u'<eos>']
            tokens += len(words)
            for word in words:
                ids.append(self.dictionary.add_word(word))

        return np.array(ids)


class PTBCorpus(Corpus):
    def __init__(self, dataroot='data/penn'):
        train = open(os.path.join(dataroot, 'train.txt')).readlines()
        valid = open(os.path.join(dataroot, 'valid.txt')).readlines()
        super(PTBCorpus, self).__init__(train, valid)


class VQADataset(object):
    def __init__(self, dataroot='data'):
        self.train_q = json.load(
            open(os.path.join(dataroot,'v2_OpenEnded_mscoco_train2014_questions.json'))
        )['questions']
        self.valid_q = json.load(
            open(os.path.join(dataroot, 'v2_OpenEnded_mscoco_val2014_questions.json'))
        )['questions']

    def get_raw_questions(self, subset):
        assert subset in ['train', 'valid']
        qset = self.train_q if subset == 'train' else self.valid_q
        raw_questions = [entry['question'] for entry in qset]
        return raw_questions


if __name__ == '__main__':
    dset = VQADataset()
    corpus = Corpus(dset.get_raw_questions('train'), dset.get_raw_questions('valid'))
