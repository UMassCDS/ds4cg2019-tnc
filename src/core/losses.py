import torch
import torch.nn as nn
from torch.autograd import Variable
from src.utils.util import log

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average


class MultiClassFocalLoss(FocalLoss):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(MultiClassFocalLoss, self).__init__(gamma, alpha, size_average)
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0), input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = nn.functional.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        wt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)

            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -(1 - wt).pow(self.gamma) * logpt

        if self.size_average:
            return loss.mean()
        return loss.sum()


class BinaryClassFocalLoss(FocalLoss):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(BinaryClassFocalLoss, self).__init__(gamma, alpha, size_average)
        if not (alpha is None or isinstance(alpha, (float, int))):
            log.error('Wrong alpha is given'); exit()

    def forward(self, input, target):
        pred = torch.sigmoid(input).view(-1)
        target = target.view(-1).long()

        pt = torch.where(torch.eq(target, 1), pred, 1-pred)
        logpt = torch.log(pt)
        wt = Variable(pt.data)

        if self.alpha is not None:
            at = torch.where(torch.eq(target, 1), self.alpha * torch.ones_like(pred),
                             (1-self.alpha) * torch.ones_like(pred))
            logpt = logpt * Variable(at)

        loss = -(1 - wt).pow(self.gamma) * logpt

        if self.size_average:
            return loss.mean()
        return loss.sum()
