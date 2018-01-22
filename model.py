import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var

import constants


# module for childsumtreelstm
class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)

    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(
                self.fh(child_h) +
                self.fx(inputs).repeat(len(child_h), 1)
            )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))
        return c, h

    def forward(self, tree, inputs):
        _ = [self.forward(tree.children[idx], inputs) for idx in range(tree.num_children)]

        if tree.num_children == 0:
            child_c = Var(inputs[0].data.new(1, self.mem_dim).fill_(0.))
            child_h = Var(inputs[0].data.new(1, self.mem_dim).fill_(0.))
        else:
            child_c, child_h = zip(* map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h)
        return tree.state


# putting the whole model together
class TreeModel(nn.Module):
    def __init__(self, vocab_size, in_dim, mem_dim, num_classes, sparsity, freeze):
        super(SimilarityTreeLSTM, self).__init__()
        
        self.sent_emb = nn.Embedding(vocab_size, in_dim, padding_idx=constants.PAD, sparse=sparsity)
        self.arb_emb = nn.Embedding(4, 4, padding_idx=constants.PAD, sparse=sparsity)
        
        if freeze:
            self.sent_emb.weight.requires_grad = False
        self.arb_emb.weight.requires_grad = False  # default freeze one-hot

        self.childsumtreelstm = ChildSumTreeLSTM(in_dim, mem_dim)
        self.FC = nn.Linear(mem_dim, num_classes)

    def forward(self, tree, sent_input, arb_input):
        sent_emb_input = self.sent_emb(sent_input)
        arb_emb_input = self.arb_emb(arb_input)

        cat_input = torch.cat((sent_emb_input, arb_emb_input), dim=1)
        state, hidden = self.childsumtreelstm(tree, cat_input)
        
        output = F.log_softmax(self.FC(state))

        return output
