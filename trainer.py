from tqdm import tqdm

import torch
from torch.autograd import Variable as Var

from utils import map_label_to_target


class Trainer(object):
    def __init__(self, args, model, criterion, optimizer):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epoch = 0
        self.step = 0

    # helper function for training
    def train(self, dataset):
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0.0
        indices = torch.randperm(len(dataset))

        for idx in tqdm(range(len(dataset)),desc='Training epoch ' + str(self.epoch + 1) + ''):
            tree, sent, arb = dataset[indices[idx]]
            
            self.step += 1
            _, loss = self._forward(tree, sent, arb, True)
            
            total_loss += loss

            if self.step % self.args.batchsize == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            self.step += 1
            _, loss = self._forward(tree, sent, arb, False)
            total_loss += loss

            if self.step % self.args.batchsize == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        # end for dataset
        
        self.epoch += 1
        return total_loss / len(dataset)

    # helper function for testing
    def test(self, dataset):
        self.model.eval()
        total_loss = 0.0
        predictions = torch.zeros(len(dataset))
        indices = torch.arange(0, dataset.num_classes)  # start from 0

        for idx in tqdm(range(len(dataset)), desc='Testing epoch  ' + str(self.epoch) + ''):
            tree, sent, arb_batch, label_batch = dataset[idx]

            output, loss = self._forward(tree, sent, arb, False)
            total_loss += loss

            # get prediction
            output = output.data.squeeze().cpu()
            predictions[idx] = torch.dot(indices, torch.exp(output))
        # end for dataset
            
        return total_loss / len(dataset), predictions

    def _forward(self, tree, sent, arb, flag):
        sent_len = sent.size()[0]
        sent_input = Var(sent)

        if flag:
            a, r, b = arb[0], arb[1], arb[2]
            target = Var(map_label_to_target(1, 2))
        else:
            a, r, b = arb[2], arb[1], arb[0]
            target = Var(map_label_to_target(0, 2))
        arb_input = self._encode_arb(a, r, b, sent_len)
            
        if self.args.cuda:
            sent_input = sent_input.cuda()
            arb_input = arb_input.cuda()
            target = target.cuda()
        
        output = self.model(tree, sent_input, arb_input)
        loss = self.criterion(output, target)
        _loss = loss.data[0]
        loss.backward()

        tree.clear_state()
        return output, _loss

    def _encode_arb(self, a, r, b, sent_len):
        arb = torch.zeros(sent_len, 3)
        for i in a:
            arb[i][0] = 1  # start from 0 
        for i in r:
            arb[i][1] = 1
        for i in b:
            arb[i][2] = 1
        return Var(arb)
