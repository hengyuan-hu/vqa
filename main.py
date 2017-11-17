import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np

from dataset import Dictionary, VQAFeatureDataset, VQAFilteredDataset
from modules import base_model
from modules import relation_model
from modules import det_model
from train import train
import utils
from collections import OrderedDict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='dev', help='dev or train?')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_hid', type=int, default=512)
    parser.add_argument('--model', type=str, default='baseline0')
    # parser.add_argument('--log', type=str, default='logs/exp0.txt')
    parser.add_argument('--output', type=str, default='saved_models/exp0')
    parser.add_argument('--batch_size', type=int, default=512)
    # parser.add_argument('--lr', type=float, default=1e-3)
    # parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    if args.task == 'dev':
        train_dset = VQAFeatureDataset('dev', dictionary)
        eval_dset = VQAFeatureDataset('dev', dictionary)
        batch_size = 100
        args.epochs = 50
    elif args.task == 'dev2':
        train_dset = VQAFeatureDataset('val', dictionary)
        eval_dset = train_dset
        batch_size = args.batch_size
    elif args.task == 'train':
        train_dset = VQAFeatureDataset('train', dictionary)
        eval_dset = VQAFeatureDataset('val', dictionary)
        batch_size = args.batch_size
    else:
        assert False, args.task

    func_name = 'build_%s' % args.model
    if 'baseline' in args.model:
        model = getattr(base_model, func_name)(train_dset, args.num_hid).cuda()
    elif 'rm' in args.model:
        model = getattr(relation_model, func_name)(train_dset, args.num_hid).cuda()
    elif 'det' in args.model:
        model = getattr(det_model, func_name)(train_dset, args.num_hid).cuda()
    else:
        assert False, 'invalid'

    # seems not necessary
    # utils.init_net(model, None)
    model.w_emb.init_embedding('data/glove6b_init_300d.npy')

    if args.task != 'dev':
        model = nn.DataParallel(model).cuda()

    spatial_dset = VQAFilteredDataset(eval_dset, utils.spatial_filter)
    action_dset = VQAFilteredDataset(eval_dset, utils.action_filter)

    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=1)
    eval_loader =  DataLoader(eval_dset, batch_size, shuffle=True, num_workers=1)
    action_loader = DataLoader(action_dset, batch_size, shuffle=True, num_workers=1)
    spatial_loader = DataLoader(spatial_dset, batch_size, shuffle=True, num_workers=1)
    eval_loaders = OrderedDict([
        ('eval', eval_loader),
        ('action', action_loader),
        ('spatial', spatial_loader),
    ])

    train(model, train_loader, eval_loaders, args.epochs, args.output)
