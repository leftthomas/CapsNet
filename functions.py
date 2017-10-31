import torch
import torch.nn as nn
from torch.autograd import Variable


class DigitMarginLoss(nn.Module):
    def __init__(self, m_plus=0.9, m_minus=0.1, lamda=0.5):
        super(DigitMarginLoss, self).__init__()
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lamda = lamda

    def forward(self, output, target):
        norm = output.norm(dim=0)
        zero = Variable(torch.zeros(1))
        losses = [torch.max(zero, self.m_plus - norm).pow(2) if digit == target.data[0]
                  else self.lamda * torch.max(zero, norm - self.m_minus).pow(2)
                  for digit in range(10)]
        return torch.cat(losses).sum()
