import os
import json
import cPickle
import numpy as np
import utils
import h5py
import torch
from torch.utils.data import Dataset
import torchvision


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                tokens.append(self.word2idx[w])
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print 'dictionary dumped to %s' % path

    @classmethod
    def load_from_file(cls, path):
        print 'loading dictionary from %s' % path
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

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


def _create_entry(img, question, answer):
    answer.pop('image_id')
    answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'image'       : img,
        'question'    : question['question'],
        'answer'      : answer}
    return entry


def _load_dataset(dataroot, name, img_id2val):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    question_path = os.path.join(
        dataroot, 'v2_OpenEnded_mscoco_%s2014_questions.json' % name)
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
    # answer_path = os.path.join(
    #     dataroot, 'v2_mscoco_%s2014_annotations.json' % name)
    # answers = sorted(json.load(open(answer_path))['annotations'],
    #                  key=lambda x: x['question_id'])
    answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
    answers = cPickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])

    utils.assert_eq(len(questions), len(answers))
    entries = []
    for question, answer in zip(questions, answers):
        utils.assert_eq(question['question_id'], answer['question_id'])
        utils.assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']
        entries.append(_create_entry(img_id2val[img_id], question, answer))

    return entries


class VQADataset(Dataset):

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # TODO: note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens


class VQAImageDataset(VQADataset):
    def __init__(self, name, img_size, dictionary,
                 # img_transform=None,
                 question_transform=None,
                 answer_transform=None,
                 dataroot='data'):
        super(VQAImageDataset, self).__init__()
        assert name in ['train', 'val']

        # self.img_transform = img_transform
        self.img_transform = torchvision.transforms.Compose([
            torchvision.transforms.Scale((img_size, img_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5])
        ])
        self.question_transform = question_transform
        self.answer_transform = answer_transform
        # TODO: implement cache
        self.dictionary = dictionary

        image_folder = os.path.join(dataroot, '%s2014' % name)
        self.images = utils.load_folder(image_folder, 'jpg')
        self.img_id2idx = {}
        for idx, img in enumerate(self.images):
            img_id = int(img.split('/')[-1].split('.')[0].split('_')[-1])
            self.img_id2idx[img_id] = idx
        self.entries = _load_dataset(dataroot, name, self.img_id2idx)

    def __getitem__(self, index):
        entry = self.entries[index]
        img = utils.pil_loader(entry['image'])
        question = entry['question']
        answer = entry['answer']

        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.question_transform is not None:
            question = self.question_transform(question)
        if self.answer_transform is not None:
            answer = self.answer_transform(answer)

        return img, question, answer

    def __len__(self):
        return len(self.entries)


class VQAFeatureDataset(VQADataset):
    def __init__(self, name, dictionary, dataroot='data'):
        super(VQAFeatureDataset, self).__init__()
        assert name in ['train', 'val', 'dev']

        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.dictionary = dictionary

        if name == 'train' or name == 'val':
            self.img_id2idx = cPickle.load(
                open(os.path.join(dataroot, '%s36_imgid2idx.pkl' % name)))
            print 'loading features from h5 file'
            h5_path = os.path.join(dataroot, '%s36.hdf5' % name)
            with h5py.File(h5_path, 'r') as hf:
                self.features = np.array(hf.get('image_features'))
                # self.bboxes = np.array(hf.get('image_bb'))

            self.entries = _load_dataset(dataroot, name, self.img_id2idx)
        else:
            self.features = np.load(os.path.join(dataroot, 'dev_features.npy'))
            # self.bboxes = np.load(os.path.join(dataroot, 'dev_bboxes.npy'))
            self.entries = cPickle.load(
                open(os.path.join(dataroot, 'dev_entries.pkl')))

        self.tokenize()
        self.tensorize()
        self.v_dim = self.features.size(2)

    def tensorize(self):
        self.features = torch.from_numpy(self.features)

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]

        features = self.features[entry['image']]
        question = entry['q_token']

        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)

        return features, question, target

    def __len__(self):
        return len(self.entries)


if __name__ == '__main__':
    import time

    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    dset = VQAFeatureDataset('val', dictionary)
    dataloader = torch.utils.data.DataLoader(
        dset, batch_size=100, shuffle=True, num_workers=4, drop_last=False)

    t = time.time()
    for b, (x, y, z) in enumerate(dataloader):
        a = x
        if b >= len(dataloader):
            break
    print('time: %.2f' % (time.time() - t))
