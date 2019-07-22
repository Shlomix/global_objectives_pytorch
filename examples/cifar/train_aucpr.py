import os, sys

path = os.path.abspath(__file__ + "/../../../")
sys.path.insert(0, path)

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn
from sklearn.metrics import average_precision_score
from global_objectives.losses import AUCPRLoss
from examples.cifar.networks import vgg, custom_cnn
from examples.cifar.misc import progress_bar
from sklearn.preprocessing import label_binarize

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

best_acc = 0

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

CLASS_ID = 2

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

train_loader = torch.utils.data.DataLoader(train_set, batch_size=512,
                                           shuffle=True, num_workers=2)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=test_transforms)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=512,
                                          shuffle=False, num_workers=2)

train_set_frozen = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=True, transform=test_transforms)

train_loader_frozen = torch.utils.data.DataLoader(train_set_frozen, batch_size=512,
                                           shuffle=False, num_workers=2)

net = custom_cnn.CustomCNN()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

use_ce = False

if use_ce:
    criterion = nn.CrossEntropyLoss()
else:
    criterion = AUCPRLoss(num_labels=10, num_anchors=20, dual_factor=2.0)

params = list(net.parameters()) + list(criterion.parameters())

optimizer_net = optim.Adam(params, lr=1e-3)
scheduler_net = optim.lr_scheduler.StepLR(optimizer_net, step_size=30, gamma=0.5)


net = net.to(device)
criterion.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


def one_hot_encoding(logits, targets):
    N, C = logits.size()
    _targets = targets.unsqueeze(-1)
    if C > 1:
        labels = torch.zeros_like(logits).scatter(1, _targets.long().data, 1)
    else:
        labels = _targets
    return labels


def train(epoch):
    print('\nTrain Epoch: %d' % epoch)
    net.train()
    criterion.train()
    train_loss = 0

    scheduler_net.step()

    for batch_idx, (inputs, targets) in enumerate(train_loader):

        targets = targets.long()

        inputs, targets = inputs.to(device), targets.to(device)

        optimizer_net.zero_grad()
        outputs = net(inputs)
        if not use_ce:
            targets = one_hot_encoding(logits=outputs, targets=targets)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer_net.step()

        train_loss += loss.item()

        targets_ = label_binarize(targets.cpu().numpy(), classes=list(range(10)))

        map = average_precision_score(targets_, outputs.detach().cpu().numpy())
        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f, MAP: %.3f'
                     % (train_loss / (len(train_loader)), map))


def _test(epoch, loader, label):
    print('\nTesting {} Epoch: {}'.format(label, epoch))
    net.eval()
    criterion.eval()
    test_loss = 0
    outputs_epoch = np.array([])
    targets_epoch = np.array([])
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            targets = targets.long()

            inputs, targets = inputs.to(device), targets.to(device)

            outputs = net(inputs)
            if not use_ce:
                targets = one_hot_encoding(logits=outputs, targets=targets)

            loss = criterion(outputs, targets)

            if outputs_epoch.size != 0:
                outputs_epoch = np.concatenate((outputs_epoch, outputs.detach().cpu().numpy()))
                targets_epoch = np.concatenate((targets_epoch, targets.detach().cpu().numpy()))
            else:
                outputs_epoch = outputs.detach().cpu().numpy()
                targets_epoch = targets.detach().cpu().numpy()

            test_loss += loss.item()

            targets_ = label_binarize(targets.cpu().numpy(), classes=list(range(10)))
            map = average_precision_score(targets_, outputs.detach().cpu().numpy())

            progress_bar(batch_idx, len(loader), 'Loss: %.3f, MAP: %.3f'
                         % (test_loss / (len(loader)), map))

        targets_ = label_binarize(targets_epoch, classes=list(range(10)))
        map = average_precision_score(targets_, outputs_epoch)
        progress_bar(len(loader), len(loader), 'Loss: %.3f, MAP: %.3f'
                     % (test_loss / (len(loader)), map))


for epoch in range(0, 100):
    train(epoch)
    _test(epoch, train_loader_frozen, "train_set")
    _test(epoch, test_loader, "test_set")

