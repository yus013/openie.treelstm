from copy import deepcopy

import torch


class Metrics():
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def _to_tensor(self, predictions, labels):
        x = list()
        for pred_batch in predictions:
            for pred in pred_batch:
                x.append(pred)
        y = list()
        for label_batch in labels:
            for label in label_batch:
                y.append(label)
        y = torch.Tensor(y)
        return torch.Tensor(x), torch.Tensor(y)


    def pearson(self, predictions, labels):
        x = deepcopy(predictions)
        y = deepcopy(labels)
        x = (x - x.mean()) / x.std()
        y = (y - y.mean()) / y.std()
        return torch.mean(torch.mul(x, y))
    
    def mse(self, predictions, labels):
        x = deepcopy(predictions)
        y = deepcopy(labels)
        return torch.mean((x - y) ** 2)
    
