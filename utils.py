import os
import numpy as np
from PIL import Image
import torch.nn as nn
from collections import OrderedDict


EPS = 1e-7


def assert_eq(real, expected):
    assert real == expected, '%s (true) vs %s (expected)' % (real, expected)


def assert_array_eq(real, expected):
    assert (np.abs(real-expected) < EPS).all(), \
        '%s (true) vs %s (expected)' % (real, expected)


def load_folder(folder, suffix):
    imgs = []
    for f in sorted(os.listdir(folder)):
        if f.endswith(suffix):
            imgs.append(os.path.join(folder, f))
    return imgs


def load_imageid(folder):
    images = load_folder(folder, 'jpg')
    img_ids = set()
    for img in images:
        img_id = int(img.split('/')[-1].split('.')[0].split('_')[-1])
        img_ids.add(img_id)
    return img_ids


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def weights_init(m):
    """custom weights initialization called on net_g and net_f."""
    cname = m.__class__
    if cname == nn.Linear or cname == nn.Conv2d or cname == nn.ConvTranspose2d:
        m.weight.data.normal_(0.0, 0.02)
    elif cname == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    else:
        print '%s is not initialized.' % cname


def init_net(net, net_file):
    if net_file:
        net.load_state_dict(torch.load(net_file))
    else:
        net.apply(weights_init)


def check_entry(entry, keywords):
    q = entry['question'].lower()
    words = q[:-1].split(" ")
    for k in keywords:
        if k in words:
            return True

    return False

action_keywords = ['holding', 'throwing', 'pointing', 'kicking', 'swinging']
action_filter = lambda x: check_entry(x, action_keywords)
spatial_keywords = ['above', 'below', 'left', 'right', 'between', 'top', 'under']
spatial_filter = lambda x: check_entry(x, spatial_keywords)


class Logger(object):
    def __init__(self, output_name):
        folder = os.path.dirname(output_name)
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.log_file = open(output_name, 'w')
        # self.infos = OrderedDict()

    # def append(self, key, val):
    #     vals = self.infos.setdefault(key, [])
    #     vals.append(val)

    def log(self, msg):
        # msgs = [extra_msg]
        # for key, vals in self.infos.iteritems():
        #     msgs.append('%s: %.6f' % (key, np.mean(vals)))
        # msg = '\n'.join(msgs)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        # self.infos = OrderedDict()
        return msg
