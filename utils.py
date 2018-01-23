import os
import math

import torch

from vocab import Vocab


# loading GLOVE word vectors
# if emb_path file is found, will load that
# else will load from raw_emb_path file & save
def load_word_vectors(emb_path, vocab_path, raw_emb_path=None):

    if os.path.isfile(vocab_path):
        print("==> Vocabulary found, loading to memory")
        vocab = Vocab(vocab_path)
    else:
        raise Exception("No vocabulary file found")
    
    if os.path.isfile(emb_path):
        print("==> Embedding found, loading to memory")
        vectors = torch.load(emb_path)
    else:
        print("==> Embedding not found, preparing...")
        dim = 0
        with open(raw_emb_path) as f:
            txt_vec = f.readline().strip().split()
            dim = len(txt_vec[1:])
        vectors = torch.zeros(len(vocab), dim)
        with open(raw_emb_path) as f:
            for line in f:
                txt_vec = line.strip().split()
                wd = line[0].lower()
                vec = torch.Tensor(list(map(float, txt_vec[1:])))
                vectors[vocab.get_idx(wd)] = vec
        torch.save(vectors, emb_path)
    # end loading embedding file
    
    return vectors, vocab


# mapping from scalar to vector
def map_label_to_target(label, num_classes):
    target = torch.zeros(1, num_classes)
    target[0][label] = 1.0
    return target
