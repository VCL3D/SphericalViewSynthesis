import torch

# Computes and stores the average and current value
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = torch.tensor(0.0)
        self.avg = torch.tensor(0.0)
        self.sum = torch.tensor(0.0)
        self.count = torch.tensor(0.0)

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count