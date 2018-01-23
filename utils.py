import os
import math

import torch

from vocab import Vocab


# loading GLOVE word vectors
# if emb_path file is found, will load that
# else will load from raw_emb_path file & save
def load_word_vectors(sent_emb_path, vocab, raw_sent_emb_path=None):
    
    if os.path.isfile(sent_emb_path):
        print("==> Sentence embedding found, loading to memory")
        sent_vectors = torch.load(sent_emb_path)
    else:
        print("==> Sentence embedding not found, preparing...")
        dim = 0
        with open(raw_sent_emb_path) as f:
            txt_vec = f.readline().strip().split()
            dim = len(txt_vec[1:])
        sent_vectors = torch.zeros(len(vocab), dim)
        with open(raw_sent_emb_path) as f:
            for line in f:
                txt_vec = line.strip().split()
                if len(txt_vec) == dim + 1:
                    wd = txt_vec[0].lower()
                    if wd in vocab.wd2idx:
                        vec = torch.Tensor(list(map(float, txt_vec[1:])))
                        sent_vectors[vocab.get_idx(wd)] = vec
            # end for lines
        torch.save(sent_vectors, sent_emb_path)
    # end loading sentence embedding file

    return sent_vectors


# mapping from scalar to vector
def map_label_to_target(label, num_classes):
    target = torch.zeros(1, num_classes)
    target[0][label] = 1.0
    return target
