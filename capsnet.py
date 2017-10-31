import torch.nn as nn
import torch.nn.functional as F

from functions import Routing


class CapsNet(nn.Module):
    def __init__(self, with_reconstruction=True):
        super(CapsNet, self).__init__()
        self.with_reconstruction = with_reconstruction

        self.conv1 = nn.Conv2d(1, 256, 9)
        self.primary_caps = nn.Conv2d(256, 32, 9, stride=2)
        self.digit_caps = Routing(4 * 6 * 6, 10, 8, 16, 4)

        if with_reconstruction:
            self.fc1 = nn.Linear(160, 512)
            self.fc2 = nn.Linear(512, 1024)
            self.fc3 = nn.Linear(1024, 784)

    def forward(self, input, target):
        conv1 = self.conv1(input)
        relu = F.relu(conv1)
        primary_caps = self.primary_caps(relu)
        digit_caps = self.digit_caps(primary_caps)

        if self.with_reconstruction:
            mask = Variable(torch.zeros(digit_caps.size()))
            mask[:, target.data[0]] = digit_caps[:, target.data[0]]
            fc1 = F.relu(self.fc1(mask.view(-1)))
            fc2 = F.relu(self.fc2(fc1))
            reconstruction = F.sigmoid(self.fc3(fc2))

        return digit_caps, reconstruction


if __name__ == '__main__':
    import torch
    from torch.autograd import Variable

    net = CapsNet()
    x = torch.rand(1, 1, 28, 28)
    net(Variable(x), Variable(torch.LongTensor(3)))
