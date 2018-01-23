import os
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.utils.data as data

import constants
from tree import Tree


# Dataset class
class ERDataset(data.Dataset):  # entity and relation 
    def __init__(self, path, vocab, num_classes):
        super(ERDataset, self).__init__()
        self.vocab = vocab
        self.num_classes = num_classes

        self.sentences = self.read_sentences(os.path.join(path, 'sent/sent.toks'))

        self.trees = self.read_trees(os.path.join(path, 'sent/sent.parents'))

        self.arbs = self.read_arbs(os.path.join(path, 'arb/arb.txt'))
        
        self.labels = self.read_labels(os.path.join(path, 'arb/label.txt'))

        self.size = len(self.labels)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        tree = deepcopy(self.trees[index])
        sent = deepcopy(self.sentences[index])
        arb_batch = deepcopy(self.arbs[index])
        label_batch = deepcopy(self.labels[index])
        return tree, sent, arb_batch, label_batch

    def read_sentences(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
            sentences = [self.read_sentence(line) for line in tqdm(lines)]
        return sentences

    def read_sentence(self, line):
        indices = self.vocab.convert_to_idxs(line.split())
        return torch.IntTensor(indices)

    def read_trees(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
            trees = [self.read_tree(line) for line in tqdm(lines)]
        return trees

    def read_tree(self, line):
        parents = list(map(int, line.split()))

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
        arbs = list()
        
        prev_sent_id = -1
        arb_batch = None

        for line in open(filename):
            sent_id, a, r, b = line.strip().split()
            if int(sent_id) != prev_sent_id:
                prev_sent_id = sent_id
                arbs.append(arb_batch)
                arb_batch = list()
            a = list(map(int, a.split(',')))
            r = list(map(int, r.split(',')))
            b = list(map(int, b.split(',')))
            arb_batch.append((a, r, b))
        # end open arb file
        return arbs

    def read_labels(self, filename):
        labels = list()
        
        prev_sent_id = -1
        label_batch = None

        for line in open(filename):
            sent_id, label = line.strip().split()
            if int(sent_id) != prev_sent_id:
                prev_sent_id = sent_id
                labels.append(label_batch)
                label_batch = list()
            label_batch.append(label)
        return labels
