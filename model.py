from sklearn.model_selection import train_test_split
import os
from glob import glob
import pandas as pd
from torchvision import transforms
import os
from glob import glob
import timm
file_path = './emotion/*/*.png'

file_list = glob(file_path)
file_list

data_dict = {'image_name':[],'class':[],'target':[], 'file_path':[]}
#target_dict = {'Angry':0,"Disgust":1,"Fear":2,"Happy":3,"Netural":4,"Sad":5,"Surprise":6}
#target_dict = {'angry':0,"Surprise":1}
target_dict = {'angry':0,'Contempt':1,"Disgust":2,"Fear":3,"Happy":4,"Sadness":5,"Surprise":6}

for path in file_list:
    data_dict['file_path'].append(path)  # file_path 항목에 파일 경로 저장

    path_list = path.split(os.path.sep)  # os별 파일 경로 구분 문자로 split
    print(path_list)
    data_dict['image_name'].append(path_list[-1])
    data_dict['class'].append(path_list[-2])
    data_dict['target'].append(target_dict[path_list[-2]])

train_df = pd.DataFrame(data_dict)
print('\n<data frame>\n', train_df)

train_df.to_csv("./emotion.csv", mode='w')
def get_df():
    # cvs 파일 읽어서 DataFrame으로 저장
    df = pd.read_csv('./emotion.csv')

    # 데이터셋을 train, val, test로 나누기
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=2359)
    df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=2359)
    df_train.resize()

    return df_train, df_val, df_test

df_train, df_val, df_test = get_df()

import torch
from torch.utils.data import Dataset
from PIL import Image


class Classification_Dataset(Dataset):
    def __init__(self, csv, mode, transform=None):
        self.csv = csv.reset_index(drop=True)  # random으로 섞인 데이터의 인덱스를 reset 시켜서 다시 부여한다.
        self.transform = transform

    def __len__(self):
        return self.csv.shape[0]  # csv 파일의 행 개수 == 데이터 개수

    def __getitem__(self, index):
        row = self.csv.iloc[index]  # 주어진 index에 대한 데이터 뽑아오기
        image = Image.open(row.file_path).convert('RGB')  # 파일경로로 부터 이미지를 읽고 rgb로 변환하기
        target = torch.tensor(self.csv.iloc[index].target).long()

        if self.transform:
            image = self.transform(image)  # 이미지에 transform 적용하기

        return image, target  # 이미지와 target return하기

import numpy as np
dataset_train = Classification_Dataset(df_train, 'train', transform=transforms.ToTensor())
rgb_mean = [np.mean(x.numpy(), axis=(1,2)) for x,_ in dataset_train]
rgb_std = [np.std(x.numpy(), axis=(1,2)) for x,_ in dataset_train]
c_mean = []
c_std = []
for i in range(3):
    c_mean.append(np.mean([m[i] for m in rgb_mean]))
    c_std.append(np.std([s[i] for s in rgb_std]))
print(c_mean,c_std)
from torchvision import transforms


def get_transforms(image_size):
    transforms_train = transforms.Compose([
        #transforms.Resize([256 , 256]),
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.50655943, 0.50655943, 0.50655943],
                             [0.05356207, 0.05356207, 0.05356207])])

    transforms_val = transforms.Compose([transforms.Resize(image_size + 30),
                                         transforms.CenterCrop(image_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.510635, 0.510635, 0.510635], [0.05335634, 0.05335634, 0.05335634]
                                                             )])

    return transforms_train, transforms_val

transforms_train, transforms_val = get_transforms(224)
dataset_train = Classification_Dataset(df_train, 'train', transform=transforms_train)
dataset_val = Classification_Dataset(df_val, 'valid', transform=transforms_val)

from torch.utils.data.sampler import RandomSampler
from torch.utils.data import DataLoader

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=10, sampler=RandomSampler(dataset_train), num_workers=0)
valid_loader = torch.utils.data.DataLoader(dataset_val, batch_size=10, num_workers=0)


from torchvision import models
from collections import OrderedDict
import torch.nn as nn
model = models.vgg16(pretrained=True)
# model = models.efficientnet_b0(pretrained=True)
# model = timm.create_model('efficientnet_b0',pretrained=True,num_classes=7)
    # Backprop을 수행하지 않도록 parameter들을 동결시키기
    # 재학습을 위해, 모든 parameters의 gradient를 꺼놓기
for param in model.parameters():
    param.requires_grad = False

#마지막 layer를 과제에 맞게 수정하기
classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 500)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(500, 7))
    ]))

model.classifier = classifier

import numpy as np
import cv2
import random
import time
import torch.optim as optim

from tqdm import tqdm


def train_epoch(model, loader, device, criterion, optimizer):
    model.train()  # 모델 train 모드로 바꾸기
    train_loss = []
    bar = tqdm(loader)
    for i, (data, target) in enumerate(bar):
        optimizer.zero_grad()  # 최적화된 모든 변수 초기화

        data, target = data.to(device), target.to(device)  # 지정한 device로 데이터 옮기기
        logits = model(data)  # 1. forward pass

        loss = criterion(logits, target)  # 2. loss 계산
        loss.backward()  # 3. backward pass

        optimizer.step()  # 4. gradient descent(파라미터 업데이트)

        loss_np = loss.detach().cpu().numpy()  # loss값 가져오기 위해 gpu에 있던 데이터 모두 cpu로 옮기기
        train_loss.append(loss_np)
        bar.set_description('loss: %.5f' % (loss_np))

    train_loss = np.mean(train_loss)  # 한 epoch당 train loss의 평균 구하기
    return train_loss


def val_epoch(model, loader, device, criterion):

    model.eval()  # 모델 evaluate 모드로 바꾸기
    val_loss = []
    LOGITS = []
    PROBS = []
    TARGETS = []
    with torch.no_grad():
        for (data, target) in tqdm(loader):
            data, target = data.to(device), target.to(device)  # 지정한 device로 데이터 옮기기
            logits = model(data)  # 1. forward pass
            probs = logits.softmax(1)  # 다중분류 -> 각 클래스일 확률을 전체 1로 두고 계산하기

            LOGITS.append(logits.detach().cpu())
            PROBS.append(probs.detach().cpu())
            TARGETS.append(target.detach().cpu())

            loss = criterion(logits, target)  # 2. loss 계산
            val_loss.append(loss.detach().cpu().numpy())

    val_loss = np.mean(val_loss)
    LOGITS = torch.cat(LOGITS).numpy()
    PROBS = torch.cat(PROBS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()

# accuracy : 정확도
    acc = (PROBS.argmax(1) == TARGETS).mean() * 100.

    return val_loss, acc


def run(model, init_lr, n_epochs):
    # gpu 사용
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model을 지정한 장치로 옮기기
    model = model.to(device)

    # loss function 지정
    criterion = nn.CrossEntropyLoss()

    # optimizer로 adam 사용
    optimizer = optim.Adam(model.parameters(), lr=init_lr)

    for epoch in range(1, n_epochs + 1):
        print(time.ctime(), f'Epoch {epoch}')

        train_loss = train_epoch(model, train_loader, device, criterion, optimizer)  # train
        val_loss, acc = val_epoch(model, valid_loader, device, criterion)  # validation

        content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid loss: {(val_loss):.5f}, Acc: {(acc):.4f}.'
        print(content)

    # torch.save(model.state_dict(), 'best_model.pth')
    torch.save(model, 'best_model_2.pth')

run(model, init_lr=4e-5, n_epochs=30)
