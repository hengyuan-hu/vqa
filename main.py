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
    # parser.add_argument('--dataset', type=str, required=True,
    #                     help='dataset to use, (PTB, VQA)')
    # parser.add_argument('--model', type=str, default='LSTM',
    #                     help='type of recurrent net (LSTM, GRU)')
    # parser.add_argument('--emsize', type=int, default=200,
    #                     help='size of word embeddings')
    # parser.add_argument('--nhid', type=int, default=200,
    #                     help='# hidden units per layer')
    # parser.add_argument('--nlayers', type=int, default=2)
    # parser.add_argument('--lr', type=float, default=20)
    # parser.add_argument('--clip', type=float, default=0.25, help='max gradient norm')
    # parser.add_argument('--epochs', type=int, default=40)
    # parser.add_argument('--batch_size', type=int, default=20)
    # parser.add_argument('--bptt', type=int, default=35, help='sequence length')
    # parser.add_argument('--dropout', type=float, default=0.2,
    #                     help='dropout applied to layers (0 = no dropout)')
    # parser.add_argument('--tied', action='store_true',
    #                     help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    # parser.add_argument('--save', type=str,  default='model.pt',
    #                     help='path to save the final model')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Set the random seed manually for reproducibility.
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    train_dset = VQAFeatureDataset('val', dictionary)
    eval_dset = VQAFeatureDataset('dev', dictionary)
    # train_dset = VQAFeatureDataset('train', dictionary)
    # eval_dset = VQAFeatureDataset('val', dictionary)

    model = base_model.build_baseline0(train_dset).cuda()
    # seems not necessary
    # utils.init_net(model, None)
    model.q_emb.init_embedding('data/glove6b_init_300d.npy')

    train(model, train_dset, train_dset, 50)
