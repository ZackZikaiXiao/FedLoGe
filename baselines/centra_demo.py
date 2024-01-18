# -*- coding:utf-8 -*-
# @Time : 2023-01-10 19:08
# @Author : DaFuChen
# @File : CSDN写作代码笔记
# @software: PyCharm
 
 
 
import torchvision
 
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from model.model_res import ResNet18, ResNet34, ResNet50
 
# 训练的次数
epoch = 2
 
# 训练的批次大小
batch_size = 4
 
# 数据集的分类类别数量
CIFAR100_class = 100
 
# 模型训练时候的学习率大小

 
def main():
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

 
    data_transform = {
        # 进行数据增强的处理
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
 
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    }
 
    train_dataset = torchvision.datasets.CIFAR100(root='./data/CIFAR100', train=True,
                                                 download=True, transform=data_transform["train"])
 
    val_dataset = torchvision.datasets.CIFAR100(root='./data/CIFAR100', train=False,
                                               download=False, transform=data_transform["val"])
 
    train_num = len(train_dataset)
    val_num = len(val_dataset)
    print("using {} images for training, {} images for validation.".format(train_num, val_num))
 
    # batch_size = batch_size
 
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
 
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw,
                                               )
 
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=nw,
                                             )
 

    model = ResNet34(100)
    # 加载预训练的权重
    state_dict = torch.load("/home/zikaixiao/zikai/aapfl/pfed_lastest/demo.pth")
    # 将加载的权重加载到模型中
    model.load_state_dict(state_dict)
    model = model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    best_acc = 0.0
 
    # 记录训练产生的数据
    train_acc_list = []
    train_loss_list = []
    val_acc_list = []
 
    for epoch in range(500):
        # train
        model.train()
        running_loss_train = 0.0
        train_accurate = 0.0
        train_bar = tqdm(train_loader)
        for images, labels in train_bar:
            optimizer.zero_grad()
 
            outputs = model(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
 
            predict = torch.max(outputs, dim=1)[1]
            train_accurate += torch.eq(predict, labels.to(device)).sum().item()
            running_loss_train += loss.item()
 
        train_accurate = train_accurate / train_num
        running_loss_train = running_loss_train / train_num
        train_acc_list.append(train_accurate)
        train_loss_list.append(running_loss_train)
 
        print('[epoch %d] train_loss: %.7f  train_accuracy: %.3f' %
              (epoch + 1, running_loss_train, train_accurate))
 
        # validate
        model.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_loader = tqdm(val_loader)
            for val_data in val_loader:
                val_images, val_labels = val_data
                outputs = model(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
 
        val_accurate = acc / val_num
        val_acc_list.append(val_accurate)
        print('[epoch %d] val_accuracy: %.3f' %
              (epoch + 1, val_accurate))
        # writer_into_excel_onlyval(save_path, train_loss_list, train_acc_list, val_acc_list, "CIFAR100")
 
        # 选择最好的模型进行保存，此处的评价指标是acc
        save_model = "./demo.pth"
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(model.state_dict(), save_model)
 
 
if __name__ == '__main__':
    main()