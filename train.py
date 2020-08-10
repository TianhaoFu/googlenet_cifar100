import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR

#CIFAR100 dataset path (python version)
#CIFAR100_PATH = '/nfs/private/cifar100/cifar-100-python'

#mean and std of cifar100 dataset
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

#CIFAR100_TEST_MEAN = (0.5088964127604166, 0.48739301317401956, 0.44194221124387256)
#CIFAR100_TEST_STD = (0.2682515741720801, 0.2573637364478126, 0.2770957707973042)

#directory to save weights file
CHECKPOINT_PATH = 'checkpoint'

#total training epoches
EPOCH = 200
MILESTONES = [60, 120, 160]

#initial learning rate
#INIT_LR = 0.1

#time of we run the script
TIME_NOW = datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss')

#tensorboard log dir
LOG_DIR = 'runs'

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10


CHECKPOINT_PATH = 'checkpoint'

def train(epoch):

    warm = 1 
    b = 32

    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        if epoch <= warm:
            warmup_scheduler.step()

        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)
        print('Training Epoch: {epoch} [{trained_samples} / {total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            # trained_samples=batch_index * b + len(images),
            trained_samples=batch_index * b + 32,
            total_samples=len(cifar100_training_loader.dataset)
        ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(epoch):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in cifar100_test_loader:

        images = images.cuda()
        labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    print('GPU INFO.....')
    print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset),
        finish - start
    ))
    print()

    #add informations to tensorboard
    writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
    writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)

    return correct.float() / len(cifar100_test_loader.dataset)

if __name__ == '__main__':

    net = get_network()

    #data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        CIFAR100_TRAIN_MEAN,
        CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=32,
        shuffle=True
    )

    cifar100_test_loader = get_test_dataloader(
        CIFAR100_TRAIN_MEAN,
        CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=32,
        shuffle=True
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.Adam(net.parameters(),lr=0.01, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * 1)

    #use tensorboard
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(
            LOG_DIR, 'googlenet', TIME_NOW))
    input_tensor = torch.Tensor(1, 3, 32, 32).cuda()
    writer.add_graph(net, input_tensor)

    #create checkpoint folder to save model
    checkpoint_path = CHECKPOINT_PATH
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    # del str
    try:
        for epoch in range(1, EPOCH):
            print("----------------开始训练第"+str(epoch)+"个epoch------------------")
            if epoch > 1:
              # 1代表warm
                train_scheduler.step(epoch)

            train(epoch)
            acc = eval_training(epoch)
            with open(r'/content/sample_data/log.txt','a') as f1:
                strs = '/nepoch: '+str(epoch) + 'acc: ' +str(acc) + '\n'
                f1.write(strs) 
            f1.close()
            #start to save best performance model after learning rate decay to 0.01
            if epoch > MILESTONES[1] and best_acc < acc:
                torch.save(net.state_dict(), checkpoint_path.format(net=net, epoch=epoch, type='best'))
                best_acc = acc
                continue
             
            if not epoch % SAVE_EPOCH:
                torch.save(net.state_dict(), 'googlenet.pth')
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print("Saved interrupt")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

    writer.close()