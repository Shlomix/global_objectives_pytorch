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
from global_objectives.utils import one_hot_encoding
from examples.cifar.networks import vgg, custom_cnn
from examples.cifar.misc import progress_bar
from sklearn.preprocessing import label_binarize
from examples.cifar.cifar_10_data_loader import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

best_acc = 0

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

CLASS_ID = 2


train_loader, val_loader = get_train_valid_loader(data_dir='./data', batch_size=512, augment=True, random_seed=42,)
test_loader = get_test_loader(data_dir='./data', batch_size=512)


#net = vgg.VGG11(num_classes=10)
net = custom_cnn.CustomCNN()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

use_ce = True

if use_ce:
    criterion = nn.BCEWithLogitsLoss()
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
        outputs = net(inputs).squeeze()
        if use_ce:
            targets = one_hot_encoding(logits=outputs, targets=targets)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer_net.step()

        train_loss += loss.item()

        targets_ = label_binarize(targets.cpu().numpy(), classes=list(range(10)))

        map = average_precision_score(targets_, outputs.detach().cpu().numpy())
        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f, MAP: %.3f'
                     % (train_loss / (len(train_loader)), map))


def test(epoch, loader, label):
    global best_map

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

            outputs = net(inputs).squeeze()
            if use_ce:
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
    test(epoch, train_loader, "train_set")
    test(epoch, val_loader, "val_set")
    test(epoch, test_loader, "test_set")

