from tqdm import tqdm

import torch
from torch.autograd import Variable as Var

from utils import map_label_to_target


class Trainer(object):
    def __init__(self, args, model, criterion, optimizer):
        super(Trainer, self).__init__()
        self.args       = args
        self.model      = model
        self.criterion  = criterion
        self.optimizer  = optimizer
        self.epoch      = 0

    # helper function for training
    def train(self, dataset):
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0.0
        indices = torch.randperm(len(dataset))

        step = 0
        for idx in tqdm(range(len(dataset)),desc='Training epoch ' + str(self.epoch + 1) + ''):
            tree, sent, arb_batch, label_batch = dataset[indices[idx]]
            sent_input = Var(sent)
            
            for i in range(len(arb_batch)):
                step += 1
                label = label_batch[i]
                target = Var(map_label_to_target(label, 2))
                
                a, r, b =  arb_batch[i]
                arb = torch.IntTensor(1, )
                arb_input = None  # TODO: encode arb
                
                if self.args.cuda:
                    sent_input = sent_input.cuda()
                    arb_input = arb_input.cuda()
                    target = target.cuda()

                output = self.model(tree, sent_input, arb_input)
                loss = self.criterion(output, target)

                total_loss += loss.data[0]
                loss.backward()

                if step % self.args.batchsize == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                tree.clear_state()
            # end for arb and label batch
        # end for dataset
        
        self.epoch += 1
        return total_loss / len(dataset)

    # helper function for testing
    def test(self, dataset):
        self.model.eval()
        total_loss = 0
        # predictions = torch.zeros(len(dataset)) TODO: wait for further update
        indices = torch.arange(1, dataset.num_classes + 1)

        for idx in tqdm(range(len(dataset)),desc='Testing epoch  ' + str(self.epoch) + ''):
            tree, sent, arb_batch, label_batch = dataset[idx]
            sent_input = Var(sent, volatile=True)

            for i in range(len(arb_batch)):
                label = label_batch[i]
                target =  Var(map_label_to_target(label, 2), volatile=True)

                a, r, b = arb_batch[i]
                arb_input = None  # TODO: encode arb

                if sent_input.args.cuda:
                    sent_input = sent_input.cuda()
                    arb_input = arb_input.cuda()
                    target = target.cuda()

                output = sent_input.model(tree, sent_input, arb_input)
                loss = self.criterion(output, target)
                total_loss += loss.data[0]

                tree.clear_state()
            # end for arb and label batch
        # end for dataset
            
        return total_loss / len(dataset)
