import sys
sys.path.insert(0, "/home/shlomi/global_objectives_pytorch")


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
from examples.cifar.networks import vgg
from examples.cifar.misc import progress_bar
from sklearn.preprocessing import label_binarize

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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

train_loader = torch.utils.data.DataLoader(train_set, batch_size=256,
                                           shuffle=True, num_workers=2)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=test_transforms)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=512,
                                          shuffle=False, num_workers=2)

net = vgg.VGG16(num_classes=10)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


#criterion = nn.CrossEntropyLoss()
criterion = AUCPRLoss(num_classes=10, num_anchors=30)

params = list(net.parameters()) + list(criterion.parameters())

optimizer_net = optim.SGD(params, lr=1e-1)
scheduler_net = optim.lr_scheduler.StepLR(optimizer_net, step_size=15, gamma=0.5)


net = net.to(device)
criterion.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True



def train(epoch):
    print('\nTrain Epoch: %d' % epoch)
    net.train()
    train_loss = 0

    scheduler_net.step()

    outputs_epoch = np.array([])
    targets_epoch = np.array([])
    for batch_idx, (inputs, targets) in enumerate(train_loader):

        targets = targets.long()

        inputs, targets = inputs.to(device), targets.to(device)

        optimizer_net.zero_grad()
        outputs = net(inputs).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer_net.step()

        if outputs_epoch.size != 0:
            outputs_epoch = np.concatenate((outputs_epoch, outputs.detach().cpu().numpy()))
            targets_epoch = np.concatenate((targets_epoch, targets.detach().cpu().numpy()))
        else:
            outputs_epoch = outputs.detach().cpu().numpy()
            targets_epoch = targets.detach().cpu().numpy()


        train_loss += loss.item()

        targets_ = label_binarize(targets.cpu().numpy(), classes=list(range(10)))

        map = average_precision_score(targets_, outputs.detach().cpu().numpy())
        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f, MAP: %.3f'
                     % (train_loss / (len(train_loader)), map))

    targets_ = label_binarize(targets_epoch, classes=list(range(10)))
    map = average_precision_score(targets_, outputs_epoch)
    progress_bar(len(train_loader), len(train_loader), 'Loss: %.3f, MAP: %.3f'
                 % (train_loss / (len(train_loader)), map))
    return map


def test(epoch):
    global best_map

    print('\nTest Epoch: %d' % epoch)
    net.eval()
    criterion.eval()
    test_loss = 0
    outputs_epoch = np.array([])
    targets_epoch = np.array([])
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            targets = targets.long()

            inputs, targets = inputs.to(device), targets.to(device)

            outputs = net(inputs).squeeze()
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

            progress_bar(batch_idx, len(train_loader), 'Loss: %.3f, MAP: %.3f'
                         % (test_loss / (len(train_loader)), map))

        targets_ = label_binarize(targets_epoch, classes=list(range(10)))
        map = average_precision_score(targets_, outputs_epoch)
        progress_bar(len(train_loader), len(train_loader), 'Loss: %.3f, MAP: %.3f'
                     % (test_loss / (len(train_loader)), map))
    return map


for epoch in range(0, 100):
    train(epoch)
    test(epoch)
