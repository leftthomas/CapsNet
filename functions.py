import torch
import torch.nn as nn
import torch.nn.functional as F
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


def squash(vec):
    norm = vec.norm()
    norm_squared = norm ** 2
    coeff = norm_squared / (1 + norm_squared)
    return (coeff / norm) * vec


def accuracy(output, target):
    pred = output.norm(dim=0).max(0)[1].data[0]
    target = target.data[0]
    return int(pred == target)


class Routing(nn.Module):
    def __init__(self, num_in_caps, num_out_caps, in_dim, out_dim, num_shared):
        super(Routing, self).__init__()
        self.in_dim = in_dim
        self.num_shared = num_shared

        self.W = [nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_shared)]
        self.b = Variable(torch.zeros(num_out_caps, num_in_caps))

    def forward(self, x):
        # TODO: make it work for batch sizes > 1
        _, in_channels, h, w = x.size()
        assert in_channels == self.num_shared * self.in_dim

        x = x.squeeze().view(self.num_shared, -1, self.in_dim)
        # print(x.size())
        groups = x.chunk(self.num_shared)
        # print(groups[0].size())
        u = [group.squeeze().chunk(h * w) for group in groups]
        pred = [self.W[i](in_vec.squeeze()) for i, group in enumerate(u) for in_vec in group]
        pred = torch.stack([torch.stack(p) for p in pred]).view(self.num_shared * h * w, -1)
        # print(pred.size())
        c = F.softmax(self.b)
        # print(c.size())
        s = torch.matmul(c, pred)
        # print(s.size())
        v = squash(s.t())
        # print(v.size())
        self.b = torch.add(self.b, torch.matmul(pred, v))
        return v


if __name__ == '__main__':
    l = Routing(4 * 6 * 6, 10, 8, 16, 4)
    t = Variable(torch.rand(1, 32, 6, 6))
    for i in range(10):
        l(t)
