from dataset import VQAImageDataset
from model import Classifier
import cPickle
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch
# from torch.utils.data import DataLoader


def detect(model, dset, threshold, output):
    batch_size = 500
    num_batches = len(dset.features) / batch_size + 1
    labels = []
    num_cls = len(dset.id2name)
    k = dset.features.size(1)

    for i in range(num_batches):
        x = dset.features[i*batch_size : (i+1)*batch_size]
        x = Variable(x).cuda()
        logits = model(x) # [batch, k, num_cls]
        logits = logits.view(-1, num_cls)
        probs = nn.functional.softmax(logits).data#.cpu().numpy()
        _, preds = probs.max(1)
        probs = probs.cpu().numpy() # [batch*k, num_cls]
        preds = preds.cpu().numpy()
        for ps, cls_idx in zip(probs, preds):
            p = ps[cls_idx]
            if p < threshold:
                cls = dset.id2name[0]
            else:
                cls = dset.id2name[cls_idx]
            labels.append(cls)

    labels = np.array(labels).reshape(-1, k)
    np.save(output, labels)
    return labels

model = Classifier(2048, 512, 2, 81).cuda()
model.load_state_dict(torch.load('det_h512_layer2.pth'))

# val_dset = VQAImageDataset('val')
# detect(model, val_dset, 0.5, 'val_det.npy')

train_dset = VQAImageDataset('train')
detect(model, train_dset, 0.5, 'train_det.npy')
