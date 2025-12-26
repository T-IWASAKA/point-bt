# -*- coding: utf-8 -*-
"""

ToDo
- 具体的なノートに変更しておく, 241217

@author: tadahaya
"""
# path setting
## この.pyが所属するPROJECTのパス, 必須入力
## PROJECT/notebooksにこの.pyは格納している, PROJECTのパスを指定する
PROJECT_PATH = '/content/drive/MyDrive/MySrc/cli_test'

# packages installed in the current environment
import sys

from symbol import parameters
import os
import datetime
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

# original packages in src
from src import utils
from src import data_handler as dh
from src.models import MyNet

# === 基本的にタスクごとに変更 ===
# argumentの設定, 概ね同じセッティングの中で振りうる条件を設定
parser = argparse.ArgumentParser(description='CLI template')
parser.add_argument('--note', type=str, help='short note for this running')
parser.add_argument('--train', type=bool, default=True) # 学習ありか否か
parser.add_argument('--seed', type=str, default=222)
parser.add_argument('--num_epoch', type=int, default=5) # epoch
parser.add_argument('--batch_size', type=int, default=128) # batch size
parser.add_argument('--lr', type=float, default=0.001) # learning rate

args = parser.parse_args()
utils.fix_seed(seed=args.seed, fix_gpu=False) # for seed control

# setup
now = datetime.datetime.now().strftime('%H%M%S')
file = os.path.basename(__file__).split('.')[0]
DIR_NAME = PROJECT_PATH + '/results/' + file + '_' + now # for output
if not os.path.exists(DIR_NAME):
    os.makedirs(DIR_NAME)
LOGGER = utils.init_logger(file, DIR_NAME, now, level_console='debug') # for logger
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # get device

# === 基本的にタスクごとに変更 ===
def prepare_data():
    """
    データの読み込み・ローダーの準備を実施
    加工済みのものをdataにおいておくか, argumentで指定したパスから呼び出すなりしてデータを読み込む
    inference用を読み込む際のものも用意しておくと楽
    
    """
    import torchvision.transforms as transforms
    from torchvision.datasets import CIFAR10

    train_trans = transforms.Compose([
        transforms.RandomAffine([0, 30], scale=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_set = CIFAR10(root='./data', train=True, download=True, transform=train_trans)
    test_set = CIFAR10(root='./data', train=False, download=True, transform=test_trans)
    train_loader = dh.prep_dataloader(train_set, args.batch_size)
    test_loader = dh.prep_dataloader(test_set, args.batch_size)
    return train_loader, test_loader


# model等の準備
def prepare_model():
    """
    model, loss, optimizer, schedulerの準備
    argumentでコントロールする場合には適宜if文使うなり

    """
    model = MyNet(output_dim=10)
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epoch, eta_min=0
        )
    return model, criterion, optimizer, scheduler


# === 基本的に触らずでOK ===
def train_epoch(model, train_loader, test_loader, criterion, optimizer):
    """
    epoch単位の学習構成, なくとも良い
    
    """
    model.train() # training
    train_batch_loss = []
    for data, label in train_loader:
        data, label = data.to(DEVICE), label.to(DEVICE) # put data on GPU
        optimizer.zero_grad() # reset gradients
        output = model(data) # forward
        loss = criterion(output, label) # calculate loss
        loss.backward() # backpropagation
        optimizer.step() # update parameters
        train_batch_loss.append(loss.item())
    model.eval() # test (validation)
    test_batch_loss = []
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(DEVICE), label.to(DEVICE)
            output = model(data)
            loss = criterion(output, label)
            test_batch_loss.append(loss.item())
    return model, np.mean(train_batch_loss), np.mean(test_batch_loss)


def fit(model, train_loader, test_loader, criterion, optimizer, scheduler):
    """
    学習
    model, train_loss, test_loss (valid_loss)を返す
    schedulerは使わないことがあるか, その場合は適宜除外
    
    """
    train_loss = []
    test_loss = []
    for epoch in trange(args.num_epoch):
        model, train_epoch_loss, test_epoch_loss = train_epoch(
            model, train_loader, test_loader, criterion, optimizer
            )
        scheduler.step() # should be removed if not necessary
        train_loss.append(train_epoch_loss)
        test_loss.append(test_epoch_loss)
        LOGGER.info(
            f'Epoch: {epoch + 1}, train_loss: {train_epoch_loss:.4f}, test_loss: {test_epoch_loss:.4f}'
            )
    return model, train_loss, test_loss


def predict(model, dataloader):
    """
    推論
    学習済みモデルとdataloaderを入力に推論
    予測値と対応するラベル, 及びaccuracyを返す
    
    """
    model.eval()
    preds = []
    labels = []
    correct = 0.0
    total = 0
    with torch.no_grad():
        for data, label in dataloader:
            data, label = data.to(DEVICE), label.to(DEVICE)
            output = model(data)
            preds.append(output)
            labels.append(label)
            correct += (output.argmax(1) == label).sum().item()
            total += len(data)
    preds = torch.cat(preds, axis=0)
    labels = torch.cat(labels, axis=0)
    if torch.cuda.is_available():
        preds = preds.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
    return preds, labels, correct/total


if __name__ == '__main__':
    # argumentでtrainとevalを分けてevalのみでもできるようにしておくと楽
    if args.train:
        # training mode
        start = time.time() # for time stamp
        # 1. data prep
        train_loader, test_loader = prepare_data()
        LOGGER.info(
            f'num_training_data: {len(train_loader)}, num_test_data: {len(test_loader)}'
            )
        # 2. model prep
        model, criterion, optimizer, scheduler = prepare_model()
        # 3. training
        model, train_loss, test_loss = fit(
            model, train_loader, test_loader, criterion, optimizer, scheduler
            )
        utils.plot_progress(train_loss, test_loss, args.num_epoch, DIR_NAME)
        utils.summarize_model(model, next(iter(train_loader))[0], DIR_NAME)
        # 4. evaluation
        preds, labels, acc = predict(model, test_loader)
        LOGGER.info(f'accuracy: {acc:.4f}')
        # 5. save results & config
        utils.to_logger(LOGGER, name='argument', obj=args)
        utils.to_logger(LOGGER, name='loss', obj=criterion)
        utils.to_logger(
            LOGGER, name='optimizer', obj=optimizer, skip_keys={'state', 'param_groups'}
            )
        utils.to_logger(LOGGER, name='scheduler', obj=scheduler)
        LOGGER.info('elapsed_time: {:.2f} min'.format((time.time() - start)/60))
    else:
        # inference mode
        # データ読み込みをtestのみに変更などが必要
        pass