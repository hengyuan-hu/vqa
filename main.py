import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from dataset import Dictionary, VQAFeatureDataset
# from modules.language_model import RNNModel
from modules import base_model
from train import train
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='dev', help='dev or train?')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--log', type=str, default='logs/exp0.txt')
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
    else:
        train_dset = VQAFeatureDataset('train', dictionary)
        eval_dset = VQAFeatureDataset('val', dictionary)
        batch_size = 512

    logger = utils.Logger(args.log)
    model = base_model.build_baseline0(train_dset).cuda()
    # seems not necessary
    # utils.init_net(model, None)
    model.q_emb.init_embedding('data/glove6b_init_300d.npy')
    train(model, train_dset, eval_dset, args.epochs, batch_size, logger)
