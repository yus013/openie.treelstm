import os
import math

import torch

from vocab import Vocab


# loading GLOVE word vectors
# if emb_path file is found, will load that
# else will load from raw_emb_path file & save
def load_word_vectors(arb_emb_path, sent_emb_path, vocab_path, raw_sent_emb_path=None):

    if os.path.isfile(vocab_path):
        print("==> Vocabulary found, loading to memory")
        vocab = Vocab(vocab_path)
    else:
        raise Exception("No vocabulary file found")
    
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
                wd = line[0].lower()
                vec = torch.Tensor(list(map(float, txt_vec[1:])))
                sent_vectors[vocab.get_idx(wd)] = vec
        torch.save(sent_vectors, sent_emb_path)
    # end loading sentence embedding file

    if os.path.isfile(arb_emb_path):
        print("==> ARB embedding found, loading to memory")
        arb_vectors = torch.load(arb_emb_path)
    else:
        print("==> ARB embedding not found, preparing")
        
        arb_vectors = torch.zeros(4, 4)
        arb_vectors[0][0] = 1.0
        arb_vectors[1][1] = 1.0
        arb_vectors[2][2] = 1.0
        arb_vectors[3][3] = 1.0

        torch.save(arb_emb_path, arb_vectors)
    # end loading arb embedding file
    return arb_vectors, sent_vectors, vocab


# mapping from scalar to vector
def map_label_to_target(label, num_classes):
    target = torch.zeros(1, num_classes)
    target[0][label] = 1.0
    return target
