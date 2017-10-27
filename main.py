import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from dataset import Dictionary, VQAFeatureDataset
from modules import base_model
from modules import ram_model
from modules import reatt
from modules import rnn_model
from train import train
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='dev', help='dev or train?')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--log', type=str, default='logs/exp0.txt')
    parser.add_argument('--num_hid', type=int, default=512)
    parser.add_argument('--model', type=str, default='baseline0')
    # parser.add_argument('--lr', type=float, default=1e-3)
    # parser.add_argument('--batch_size', type=int, default=20)
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
        batch_size = 512
    elif args.task == 'train':
        train_dset = VQAFeatureDataset('train', dictionary)
        eval_dset = VQAFeatureDataset('val', dictionary)
        batch_size = 512
    else:
        assert False, args.task

    logger = utils.Logger(args.log)
    if args.model == 'baseline0':
        model = base_model.build_baseline0(train_dset, args.num_hid).cuda()
    elif args.model == 'baseline1':
        model = base_model.build_baseline1(train_dset, args.num_hid).cuda()
    elif args.model == 'baseline0_bidirect':
        model = base_model.build_baseline0_bidirect(train_dset, args.num_hid).cuda()
    elif args.model == 'ram0':
        model = ram_model.build_ram0(train_dset, args.num_hid).cuda()
    elif args.model == 'reatt0':
        model = reatt.build_reatt0(train_dset, args.num_hid).cuda()
    elif args.model == 'reatt2':
        model = reatt.build_reatt2(train_dset, args.num_hid).cuda()
    elif args.model == 'rnn0':
        model = rnn_model.build_rnn0(train_dset, args.num_hid).cuda()
    else:
        assert False, 'invalid'
    # seems not necessary
    # utils.init_net(model, None)
    model.q_emb.init_embedding('data/glove6b_init_300d.npy')
    train(model, train_dset, eval_dset, args.epochs, batch_size, logger)
