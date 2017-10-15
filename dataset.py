import os
import json
import torch
from torch.utils.data import Dataset
import torchvision
import numpy as np
from PIL import Image


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


# class _Image(object):
#     def __init__(self, path, questions, answers):
#         self.path = path

#         # TODO: pair-up question and answers
#         self.questions = self.questions
#         self.answers = answers


def load_folder(folder, suffix):
    imgs = []
    for f in sorted(os.listdir(folder)):
        if f.endswith(suffix):
            imgs.append(os.path.join(folder, f))
    return imgs


def _create_entries(img_path, questions, answers):
    """Take an image and all questions/answers for this image

    return a list of entry [{'image': _, 'question': _, 'answer': _}]
    """
    assert len(questions) == len(answers)
    questions = sorted(questions, key=lambda x: x['question_id'])
    answers = sorted(answers, key=lambda x: x['question_id'])
    entries = []
    for q, a in zip(questions, answers):
        assert q['question_id'] == a['question_id']
        entries.append({'image': img_path, 'question': q, 'answer': a})
    return entries


def _load_dataset(dataroot, name):
    question_path = os.path.join(
        dataroot,'v2_OpenEnded_mscoco_%s2014_questions.json' % name)
    answer_path = os.path.join(
        dataroot,'v2_mscoco_%s2014_annotations.json' % name)
    image_folder = os.path.join(dataroot, '%s2014' % name)

    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['image_id'])
    answers = sorted(json.load(open(answer_path))['annotations'],
                     key=lambda x: x['image_id'])
    images = load_folder(image_folder, 'jpg')

    entries = []
    qidx = 0
    aidx = 0
    for img_path in images:
        img_id = int(img_path.split('/')[-1].split('.')[0].split('_')[-1])
        qs = []
        ans = []
        while qidx < len(questions) and questions[qidx]['image_id'] == img_id :
            qs.append(questions[qidx])
            qidx += 1
        while aidx < len(answers) and answers[aidx]['image_id'] == img_id:
            ans.append(answers[aidx])
            aidx += 1

        if len(qs) == 0:
            print 'Warning: image %s has no questions' % img_path
        assert qidx == aidx, 'question and answer mismatch'
        entries += _create_entries(img_path, qs, ans)

    return entries


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class VQADataset(Dataset):
    def __init__(self, name, img_size,
                 # img_transform=None,
                 question_transform=None,
                 answer_transform=None,
                 dataroot='data'):
        super(VQADataset, self).__init__()
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
        self.entries = _load_dataset(dataroot, name)


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


if __name__ == '__main__':
    import time

    dset = VQADataset('train', 256)
    dataloader = torch.utils.data.DataLoader(
        dset, batch_size=100, shuffle=True, num_workers=0, drop_last=False)
    t = time.time()
    for b, (x, y, z) in enumerate(dataloader):
        a = x
        if b >= len(dataloader) / 100:
            break
    print('time: %.2f' % (time.time() - t))
    # corpus = Corpus(dset.get_raw_questions('train'), dset.get_raw_questions('valid'))
