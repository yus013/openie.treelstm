import json
import constants

class Vocab(object):
    def __init__(self, filename, labels=None):  # default lower case
        self.idx2wd = list()
        self.wd2idx = dict()

        if labels is not None:
            self._add_specials(labels)
        self._add_file(filename)
        self.update_idx()

    def __len__(self):
        return len(self.idx2wd)
    
    def _add_file(self, filename):
        with open(filename, 'r') as f:
            self.idx2wd.extend(json.load(f))
    
    def _add_specials(self, labels):
        for label in labels:
            self.idx2wd.append(label)
    
    def update_idx(self):
        self.wd2idx = dict()
        for i, wd in enumerate(self.wd2idx):
            self.wd2idx[wd] = i
    
    def get_idx(self, wd):
        return self.wd2idx.get(wd.lower(), self.wd2idx[constants.UNK_WORD])

    def get_wd(self, idx):
        return self.idx2wd[idx]

    def convert_to_idxs(self, wd_lst, BOS=None, EOS=None):
        """
        @param BOS: begin of sentence
        @param EOS: end of sentence
        """

        idx_lst = list()  # word vector

        if BOS is not None:
            idx_lst.append(self.wd2idx[BOS])
        
        for wd in wd_lst:
            idx_lst.append(self.wd2idx.get(wd.lower(), self.wd2idx[constants.UNK_WORD]))
        
        if EOS is not None:
            idx_lst.append(self.wd2idx[EOS])

        return idx_lst

    def convert_to_wds(self, idx_lst, stop=-1):
        wd_lst = list()

        for idx in idx_lst:
            wd_lst.append(self.idx2wd[idx])
            if idx == stop:
                break
        
        return wd_lst
