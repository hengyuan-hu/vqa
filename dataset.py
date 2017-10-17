import os
import json
import torch
from torch.utils.data import Dataset
import torchvision
import numpy as np
from PIL import Image
import utils
import cPickle


class Dictionary(object):
    def __init__(self, word2idx={}, idx2word=[]):
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
        dataroot,'v2_OpenEnded_mscoco_%s2014_questions.json' % name)
    answer_path = os.path.join(
        dataroot,'v2_mscoco_%s2014_annotations.json' % name)
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
    answers = sorted(json.load(open(answer_path))['annotations'],
                     key=lambda x: x['question_id'])

    utils.assert_eq(len(questions), len(answers))
    entries = []
    for qidx in range(len(questions)):
        utils.assert_eq(questions[qidx]['question_id'], answers[qidx]['question_id'])
        utils.assert_eq(questions[qidx]['image_id'], answers[qidx]['image_id'])
        img_id = questions[qidx]['image_id']
        entries.append(_create_entry(
            img_id2val[img_id], questions[qidx], answers[qidx]))

    return entries


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class VQADataset(Dataset):
    def __init__(self, name, img_size, dictionary,
                 # img_transform=None,
                 question_transform=None,
                 answer_transform=None,
                 dataroot='data'):
        super(VQADataset, self).__init__()
        # assert False, 'Deprecated'
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

    def __getitem__(self, index):
        entry = self.entries[index]
        img = pil_loader(entry['image'])
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


class VQAFeatureDataset(Dataset):
    def __init__(self, name, dictionary):
        super(VQAFeatureDataset, self).__init__()
        assert name in ['train', 'val']




if __name__ == '__main__':
    import time

    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    dset = VQADataset('val', 256, dictionary)
    dset.tokenize()

    token_data = torch.from_numpy(np.array(dset.entries[0]['q_token']))
    emb_layer = torch.nn.Embedding(
        dictionary.ntoken+1, 300, padding_idx=dictionary.padding_idx)
    emb_init = torch.from_numpy(np.load('data/glove6b_init_300d.npy'))
    print emb_init.size()
    print emb_layer.weight.data.size()
    emb_layer.weight.data[:dictionary.ntoken] = emb_init


    # dataloader = torch.utils.data.DataLoader(
    #     dset, batch_size=100, shuffle=True, num_workers=0, drop_last=False)
    # t = time.time()
    # for b, (x, y, z) in enumerate(dataloader):
    #     a = x
    #     if b >= len(dataloader) / 100:
    #         break
    # print('time: %.2f' % (time.time() - t))
    # corpus = Corpus(dset.get_raw_questions('train'), dset.get_raw_questions('valid'))
