import os
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.utils.data as data

import constants
from tree import Tree

import numpy as np
import pandas as pd

# Dataset class
class ERDataset(data.Dataset):  # entity and relation 
    def __init__(self, dataset_dir, vocab, num_classes):
        super(ERDataset, self).__init__()
        self.vocab = vocab
        self.num_classes = num_classes

        sent_path = os.path.join(dataset_dir, 'sent.pkl')
        tree_path = os.path.join(dataset_dir, 'parents.npy')
        arb_path = os.path.join(dataset_dir, 'arb.npy')

        self.sentences = self.read_sentences(sent_path)
        self.trees = self.read_trees(tree_path)
        self.arbs = self.read_arbs(arb_path)
        
        self.size = len(self.arbs)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        tree = deepcopy(self.trees[index])
        sent = deepcopy(self.sentences[index])
        arb_batch = deepcopy(self.arbs[index])
        return tree, sent, arb_batch

    def read_sentences(self, filename):
        lines = pd.read_pickle(filename)
        sentences = [self.read_sentence(_) for _ in tqdm(lines)]
        return sentences

    def read_sentence(self, line):
        indices = self.vocab.convert_to_idxs(line)
        return torch.LongTensor(indices)

    def read_trees(self, filename):
        lines = np.load(filename)
        trees = [self.read_tree(_) for _ in tqdm(lines)]
        return trees

    def read_tree(self, parents):
        trees = [Tree(idx) for idx in range(len(parents))]
        root = None

        for i, p in enumerate(parents):
            if p == 0:
                root = trees[i]
            else:
                prev = i
                while p > 0:
                    trees[p - 1].add_child(trees[prev])
                    parents[prev] = 0  # prevent adding the same child
                    prev = p - 1
                    p = parents[prev]
        # end build tree
        return root

    def read_arbs(self, filename):
        return np.load(filename)