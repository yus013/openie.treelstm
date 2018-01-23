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
    train_dir = os.path.join(args.data, 'train/')
    dev_dir = os.path.join(args.data, 'dev/')
    test_dir = os.path.join(args.data, 'test/')

    sent_train_dir = os.path.join(train_dir, 'sent/')
    sent_dev_dir = os.path.join(dev_dir, 'sent/')
    sent_test_dir = os.path.join(test_dir, 'sent/')

    arb_train_dir = os.path.join(train_dir, 'train/')
    arb_dev_dir = os.path.join(dev_dir, 'dev/')
    arb_test_dir = os.path.join(test_dir, 'test/')

    # load vocabulary
    vocab_path = os.path.join(args.data, "vocab.json")
    vocab = Vocab(
        filename=vocab_path, 
        labels=[constants.PAD_WORD, constants.UNK_WORD, constants.BOS_WORD, constants.EOS_WORD]
    )
    logger.debug('==> vocabulary size : %d ' % len(vocab))

    # load train dataset
    train_file = os.path.join(train_dir, "ERdata.dat")
    if os.path.isfile(train_file):
        train_dataset = torch.load(train_file)
    else:
        train_dataset = ERDataset(train_dir, vocab, 2)
        torch.save(train_dataset, train_file)
    logger.debug('==> train data size: %d' % len(train_dataset))

    # load dev dataset
    # TODO: under construction

    # load test dataset
    # TODO: under construction

    # trainer: 
    # tree model
    model = TreeModel(
        len(vocab),
        args.input_dim,
        args.mem_dim,
        2,  # 0-1 prediction
        args.sparse,
        args.freeze_embed
    )

    # criterion
    criterion = nn.KLDivLoss()
    if args.cuda:
        model.cuda(), criterion.cuda()

    # optimizer
    if args.optim == 'adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=args.lr, weight_decay=args.wd
        )
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=args.lr, weight_decay=args.wd
        )
    elif args.optim == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=args.lr, weight_decay=args.wd
        )
    else:
        raise Exception("Unknown optimizer")

    # metrics
    metrics = Metrics(2)  # 0-1 prediction

    # embeddings
    sent_emb_path = os.path.join(args.data, "sent_emb.dat")
    raw_sent_emb_path = os.path.join(args.glove, 'glove.840B.300d.txt')

    sent_emb = load_word_vectors(sent_emb_path, vocab, raw_sent_emb_path)
    
    logger.debug('==> sentence embedding size: %d * %d' % (sent_emb.size()[0], sent_emb.size()[1]))
    if args.cuda:
        sent_emb.cuda()
    model.sent_emb.weight.data.copy_(sent_emb)

    trainer = Trainer(args, model, criterion, optimizer)

    # train and test
    best = float("-inf")
    for epoch in range(args.epochs):
        train_loss = trainer.train(train_dataset)
        # train_loss, train_pred = trainer.test(train_dataset)
        # dev_loss, dev_pred = trainer.test(dev_dataset)
        # test_loss, test_pred = trainer.test(test_dataset)

        # train_pearson = metrics.pearson(train_pred, train_dataset.labels)
        # train_mse = metrics.mse(train_pred, train_dataset.labels)
        # logger.info('==> Epoch {}, Train \tLoss: {}\tPearson: {}\tMSE: {}'.format(epoch, train_loss, train_pearson, train_mse))
        
        # dev_pearson = metrics.pearson(dev_pred, dev_dataset.labels)
        # dev_mse = metrics.mse(dev_pred, dev_dataset.labels)
        # logger.info('==> Epoch {}, Dev \tLoss: {}\tPearson: {}\tMSE: {}'.format(epoch, dev_loss, dev_pearson, dev_mse))
        # test_pearson = metrics.pearson(test_pred, test_dataset.labels)
        # test_mse = metrics.mse(test_pred, test_dataset.labels)
        # logger.info('==> Epoch {}, Test \tLoss: {}\tPearson: {}\tMSE: {}'.format(epoch, test_loss, test_pearson, test_mse))

        # if best < dev_pearson:
        #     best = dev_pearson
        #     checkpoint = {
        #         'model': trainer.model.state_dict(), 
        #         'optim': trainer.optimizer,
        #         'pearson': dev_pearson, 'mse': dev_mse,
        #         'args': args, 'epoch': epoch
        #         }
        #     logger.debug('==> New optimum found, checkpointing everything now...')
        #     torch.save(checkpoint, '%s.pt' % os.path.join(args.save, args.expname))


if __name__ == "__main__":
    main()