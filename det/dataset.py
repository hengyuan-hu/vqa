import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import cPickle
import numpy as np
import utils
import h5py
import torch
from torch.utils.data import Dataset
import torchvision
from PIL import Image, ImageDraw


def _load_dataset(dataroot, name, id2path, id2feature, id2bboxes):
    """Load entries

    image_ids: dict {img_id -> path} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    question_path = os.path.join(
        dataroot, 'v2_OpenEnded_mscoco_%s2014_questions.json' % name)
    questions = json.load(open(question_path))['questions']
    questions = sorted(questions, key=lambda x: x['image_id'])
    answer_path = os.path.join(
        dataroot, 'v2_mscoco_%s2014_annotations.json' % name)
    answers = json.load(open(answer_path))['annotations']
    answers = sorted(answers, key=lambda x: x['question_id'])
    utils.assert_eq(len(questions), len(answers))

    image_ids = sorted(id2path.keys())
    entries = []
    i = 0
    for image_id in image_ids:
        path = id2path[image_id]
        feature_idx = id2feature[image_id]
        bboxes = id2bboxes[image_id]

        qa = []
        while i < len(questions) and questions[i]['image_id'] == image_id:
            assert answers[i]['image_id'] == image_id
            qa.append((
                questions[i]['question'],
                answers[i]['multiple_choice_answer']))
            i += 1

        img = {'image_id': image_id,
               'path': path,
               'feature_idx': feature_idx,
               'qa': qa}
        img.update(bboxes)
        entries.append(img)
    return entries


def load_annotation(name, dataroot):
    assert name in ['train', 'val']
    dataroot = os.path.join(dataroot, 'annotations')
    filename = 'instances_%s2014.json' % name
    js_obj = json.load(open(os.path.join(dataroot, filename)))
    annotations = js_obj['annotations']
    # return annotations, None, None
    annotations = sorted(annotations, key=lambda x: x['image_id'])

    image_info = js_obj['images']
    image_info = sorted(image_info, key=lambda x: x['id'])

    categories = js_obj['categories']
    id2name = {}
    for cat in categories:
        id_ = cat['id']
        name = cat['name']
        id2name[id_] = name

    return annotations, image_info, id2name


def convert_bboxes(annotations, image_info, id2name, oldid2newid):
    """Convert bboxes in annotations to [x_small, y_small, x_large, y_large ] format
    x and y are rescaled into [0, 1] depending on the weight and height

    return a dictionary that:
    {image_id -> {
        height: float,
        width: float,
        bboxes: np.array,
        category_ids: [int],
        names: ['string'],
        url: string (coco_url)
    """
    id2bboxes = {}
    anno_idx = 0
    num_anno = len(annotations)
    for img in image_info:
        img_id = img['id']
        height = float(img['height'])
        width = float(img['width'])
        url = img['coco_url']
        bboxes = []
        ids = []
        names = []
        while anno_idx < num_anno and annotations[anno_idx]['image_id'] == img_id:
            x0, y0, w, h = annotations[anno_idx]['bbox']
            x1 = x0 + w
            y1 = y0 + h
            bboxes.append([x0 / width, y0 / height, x1 / width, y1 / height])
            id_ = annotations[anno_idx]['category_id']
            new_id = oldid2newid[id_]
            ids.append(new_id)
            names.append(id2name[new_id])
            anno_idx += 1

        bboxes = np.array(bboxes)
        ids = np.array(ids)

        id2bboxes[img_id] = {
            'height': height,
            'width': width,
            'bboxes': bboxes,
            'ids': ids,
            'names': names,
            'url': url
        }
    return id2bboxes


def load_bboxes(name, dataroot):
    annotations, image_info, id2name = load_annotation(name, dataroot)

    # convert to consecutive numbers
    new_id2name = {}
    oldid2newid = {}
    old_ids = id2name.keys()
    for i in range(1, len(id2name)+1):
        new_id2name[i] = id2name[old_ids[i-1]]
        oldid2newid[old_ids[i-1]] = i

    id2bboxes = convert_bboxes(annotations, image_info, new_id2name, oldid2newid)
    return id2bboxes, new_id2name


def load_id2path(name, dataroot):
    path = os.path.join(dataroot, '%s_imgid2path.pkl' % name)
    if os.path.exists(path):
        print 'loading image_id -> image_path from cached file'
        id2path = cPickle.load(open(path))
        return id2path

    image_folder = os.path.join(dataroot, '%s2014' % name)
    images = utils.load_folder(image_folder, 'jpg')

    id2path = {}
    for idx, img in enumerate(images):
        id_ = int(img.split('/')[-1].split('.')[0].split('_')[-1])
        id2path[id_] = img

    print 'writing image_id -> image_path from cached file'
    cPickle.dump(id2path, open(path, 'wb'))
    return id2path


def compute_iou(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
    sa = (ax2 - ax1) * (ay2 - ay1)
    sb = (bx2 - bx1) * (by2 - by1)

    ix1 = max(ax1, bx1)
    ix2 = min(ax2, bx2)
    iy1 = max(ay1, by1)
    iy2 = min(ay2, by2)
    si = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    su = sa + sb - si
    return si / su


def _compute_detection_targets(gts, labels, bboxes, threshold):
    if len(gts) == 0:
        targets = [0 for _ in bboxes]
        return targets

    targets = []
    for bbox in bboxes:
        ious = []
        for gt in gts:
            iou = compute_iou(
                bbox[0], bbox[1], bbox[2], bbox[3], gt[0], gt[1], gt[2], gt[3])
            ious.append(iou)
        label_idx = np.argmax(ious)
        if ious[label_idx] >= threshold:
            targets.append(labels[label_idx])
        else:
            targets.append(0) # 0 means negative sample
    return targets


def compute_detection_targets(entries, spatials, threshold):
    num_imgs, num_objs, _ = spatials.shape
    targets = np.zeros((num_imgs, num_objs), dtype=np.int32)
    for entry in entries:
        gts = entry['bboxes']
        labels = entry['ids']
        bboxes = spatials[entry['feature_idx']]
        target = _compute_detection_targets(gts, labels, bboxes, threshold)
        targets[entry['feature_idx']] = np.array(target, dtype=np.int32)
    return targets


class VQAImageDataset(Dataset):
    def __init__(self, name, dataroot='../data', model=None):
        super(VQAImageDataset, self).__init__()
        assert name in ['train', 'val']

        id2path = load_id2path(name, dataroot)
        id2bboxes, self.id2name = load_bboxes(name, dataroot)
        id2feature = cPickle.load(
            open(os.path.join(dataroot, '%s36_imgid2idx.pkl' % name)))

        self.id2name[0] = '__background__'
        self.entries = _load_dataset(dataroot, name, id2path, id2feature, id2bboxes)

        print 'loading features from h5 file'
        h5_path = os.path.join(dataroot, '%s36.hdf5' % name)
        with h5py.File(h5_path, 'r') as hf:
            self.features = np.array(hf.get('image_features'))
            self.spatials = np.array(hf.get('spatial_features'))

        self.det_targets = compute_detection_targets(self.entries, self.spatials, 0.5)
        self.tensorize()
        self.model = model

    def tensorize(self):
        self.features = torch.from_numpy(self.features)
        self.det_targets = torch.from_numpy(self.det_targets).long()

    def show(self, index):
        path = self.entries[index]['path']
        width = self.entries[index]['width']
        height = self.entries[index]['height']
        image = Image.open(path)
        canvas = ImageDraw.Draw(image)

        for obj_idx in range(len(self.entries[index]['names'])):
            name = self.entries[index]['names'][obj_idx]
            x1, y1, x2, y2 = self.entries[index]['bboxes'][obj_idx]
            x1 *= width
            x2 *= width
            y1 *= height
            y2 *= height
            canvas.rectangle([x1, y1, x2, y2], outline=0)
            canvas.text([x1, y1], name)

        feature_idx = self.entries[index]['feature_idx']
        if self.model is not None:
            feature =  self.features[feature_idx].cuda()
            logits = self.model(torch.autograd.Variable(feature)).data
            _, det_cls = logits.max(1)

        for obj_idx in range(36):
            x1, y1, x2, y2 = self.spatials[feature_idx][obj_idx][:4]
            x1 *= width
            x2 *= width
            y1 *= height
            y2 *= height
            target = self.det_targets[feature_idx][obj_idx]
            if target > 0:
                canvas.rectangle([x1, y1, x2, y2], outline=255)
                if self.model is not None:
                    canvas.text([x1, y1], self.id2name[det_cls[obj_idx]])
                else:
                    canvas.text([x1, y1], self.id2name[target])

        print self.entries[index]['qa']
        image.show()

    def __getitem__(self, index):
        feature_idx = self.entries[index]['feature_idx']
        feature = self.features[feature_idx]
        target = self.det_targets[feature_idx]
        return feature, target

    def __len__(self):
        return len(self.entries)


if __name__ == '__main__':
    from model import Classifier

    model = Classifier(2048, 512, 2, 81).cuda()
    model.load_state_dict(torch.load('det_h512_layer2.pth'))

    # eval_dset = VQAImageDataset('val')
