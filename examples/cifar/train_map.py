import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn
from networks import vgg
from map_loss import AUCPR
from sklearn.metrics import average_precision_score
from misc import progress_bar
import numpy as np
best_acc = 0

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

CLASS_ID = 3  #CAT

train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=True, transform=train_transforms)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=128,
                                           shuffle=True, num_workers=2)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=test_transforms)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=128,
                                          shuffle=False, num_workers=2)

net = vgg.VGG11()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


criterion = AUCPR(pos_prior_prob=0.1, num_anchors=20)

optimizer_net = optim.SGD(net.parameters(), lr=1e-3, momentum=0.)
optimizer_b = optim.SGD(criterion.get_bs(), lr=1e-3, momentum=0.)
optimizer_lambdas = optim.SGD(criterion.get_lambdas(), lr=1e-3, momentum=0.)

scheduler_net = optim.lr_scheduler.StepLR(optimizer_net, step_size=10, gamma=0.5)
scheduler_b = optim.lr_scheduler.StepLR(optimizer_b, step_size=10, gamma=0.5)
scheduler_lambdas = optim.lr_scheduler.StepLR(optimizer_lambdas, step_size=10, gamma=0.5)

net = net.to(device)
criterion = criterion.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0

    scheduler_net.step()
    scheduler_b.step()
    scheduler_lambdas.step()

    outputs_epoch = []
    targets_epoch = []
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        targets_for_map = (targets == CLASS_ID).float()
        #targets = 2*(targets == CLASS_ID).float() - 1
        targets = targets_for_map

        targets_epoch = np.concatenate((targets_epoch, targets_for_map.detach().cpu().numpy()))

        inputs, targets = inputs.to(device), targets.to(device)

        optimizer_net.zero_grad()
        optimizer_b.zero_grad()
        optimizer_lambdas.zero_grad()

        outputs_ = net(inputs).squeeze()
        loss = criterion(outputs_, targets)
        loss.backward()
        optimizer_net.step()

        optimizer_net.zero_grad()
        optimizer_b.zero_grad()
        optimizer_lambdas.zero_grad()
        outputs_ = net(inputs).squeeze()
        loss = criterion(outputs_, targets)
        loss.backward()
        optimizer_b.step()

        optimizer_net.zero_grad()
        optimizer_b.zero_grad()
        optimizer_lambdas.zero_grad()
        outputs = net(inputs).squeeze()
        loss = -criterion(outputs, targets)
        loss.backward()
        optimizer_lambdas.step()

        outputs_epoch = np.concatenate((outputs_epoch, outputs.detach().cpu().numpy()))

        train_loss += -loss.item()

        map = average_precision_score(targets.cpu().numpy(), outputs.detach().cpu().numpy())
        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f, MAP: %.3f'
                     % (train_loss / (len(train_loader)), map))

    map = average_precision_score(targets_epoch, outputs_epoch)
    progress_bar(len(train_loader), len(train_loader), 'Loss: %.3f, MAP: %.3f'
                 % (train_loss / (len(train_loader)), map))


def test(epoch):
    global best_map
    net.eval()
    criterion.eval()
    test_loss = 0
    outputs_epoch = []
    targets_epoch = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            targets_for_map = (targets == CLASS_ID).float()
            #targets = 2 * (targets == CLASS_ID).float() - 1
            targets = targets_for_map

            targets_epoch = np.concatenate((targets_epoch, targets_for_map.detach().cpu().numpy()))

            inputs, targets = inputs.to(device), targets.to(device)

            outputs = net(inputs).squeeze()
            loss = criterion(outputs, targets)

            outputs_epoch = np.concatenate((outputs_epoch, outputs.detach().cpu().numpy()))

            test_loss += loss.item()

            progress_bar(batch_idx, len(test_loader), 'Loss: %.3f'
                         % (test_loss / (batch_idx + 1)))

        map = average_precision_score(targets_epoch, outputs_epoch)
        progress_bar(len(test_loader), len(test_loader), 'Loss: %.3f, MAP: %.3f'
                     % (test_loss / (len(test_loader)), map))


for epoch in range(0, 200):
    train(epoch)
    test(epoch)