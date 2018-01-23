from __future__ import division
from __future__ import print_function

import os
import random
import logging

import torch
import torch.nn as nn
import torch.optim as optim

# IMPORT CONSTANTS
import constants
# NEURAL NETWORK MODULES/LAYERS
from model import TreeModel
# DATA HANDLING CLASSES
from vocab import Vocab
# DATASET CLASS FOR SICK DATASET
from dataset import ERDataset
# METRICS CLASS FOR EVALUATION
from metrics import Metrics
# UTILITY FUNCTIONS
from utils import load_word_vectors
# CONFIG PARSER
from config import parse_args
# TRAIN AND TEST HELPER FUNCTIONS
from trainer import Trainer

def main():
    global args 
    args = parse_args()

    # global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
    # file logger
    fh = logging.FileHandler(os.path.join(args.save, args.expname)+'.log', mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # console logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if not torch.cuda.is_available() and args.cuda:
        args.cuda = False
        logger.info("CUDA is unavailable, convert to cpu mode")

    if args.sparse and args.wd != 0:
        logger.error('Sparsity and weight decay are incompatible, pick one!')
        exit()

    logger.debug(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # set directory
    sent_dir = os.path.join(args.data, "sent/")
    arb_dir = os.path.join(args.data, "arb/")

    sent_train_dir = os.path.join(sent_dir, 'train/')
    sent_dev_dir = os.path.join(sent_dir, 'dev/')
    sent_test_dir = os.path.join(sent_dir, 'test/')

    arb_train_dir = os.path.join(arb_dir, 'train/')
    arb_dev_dir = os.path.join(arb_dir, 'dev/')
    arb_test_dir = os.path.join(arb_dir, 'test/')

    vocab_path = os.path.join(args.data, "vocab.txt")

    vocab = Vocab(
        filename=vocab_path, 
        labels=[constants.PAD_WORD, constants.UNK_WORD, constants.BOS_WORD, constants.EOS_WORD]
    )
    logger.debug('==> vocabulary size : %d ' % len(vocab))






    




if __name__ == "__main__":
    main()