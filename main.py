import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets

from capsnet import CapsNet
from functions import DigitMarginLoss
from functions import accuracy

train_loader = DataLoader(datasets.MNIST('data', train=True, download=True, transform=transforms.Compose([
    # transforms.RandomShift(2),
    transforms.ToTensor()])), batch_size=1, shuffle=True)

test_loader = DataLoader(datasets.MNIST('data', train=False, transform=transforms.Compose([
    transforms.ToTensor()])), batch_size=1)

model = CapsNet()
optimizer = optim.Adam(model.parameters())
margin_loss = DigitMarginLoss()
reconstruction_loss = torch.nn.MSELoss(size_average=False)
model.train()

for epoch in range(1, 11):
    epoch_tot_loss = 0
    epoch_tot_acc = 0
    for batch, (data, target) in enumerate(train_loader, 1):
        data = Variable(data)
        target = Variable(target)

        digit_caps, reconstruction = model(data, target)
        loss = margin_loss(digit_caps, target) + 0.0005 * reconstruction_loss(reconstruction, data.view(-1))
        epoch_tot_loss += loss

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        acc = accuracy(digit_caps, target)
        epoch_tot_acc += acc

        template = '[Epoch {}] Loss: {:.4f} ({:.4f}), Acc: {:.2f}%'
        print(template.format(epoch, loss.data[0], (epoch_tot_loss / batch).data[0], 100 * (epoch_tot_acc / batch)))
