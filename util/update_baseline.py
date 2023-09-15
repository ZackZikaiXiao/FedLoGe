# python version 3.7.1
# -*- coding: utf-8 -*-

from cProfile import label
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
from util.util import shot_split
import copy
from util.losses import *

import sklearn.metrics as metrics
from util.losses import FocalLoss
from util.etf_methods import *
import matplotlib.pyplot as plt


# 通过cient的id来划分local longtail的数据
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        # self.loss_func = self.get_loss()  # loss function -- cross entropy
        # self.loss_func = nn.CrossEntropyLoss() # loss function -- cross entropy
        self.ldr_train, self.ldr_test = self.train_test(dataset, list(idxs))

    def get_loss(self):
        if self.args.loss_type == 'CE':
            return nn.CrossEntropyLoss()
        elif self.args.loss_type == 'focal':
            return FocalLoss(gamma=1).cuda(self.args.gpu)

    def train_test(self, dataset, idxs):
        # split training set, validation set and test set
        train = DataLoader(DatasetSplit(dataset, idxs),
                           batch_size=self.args.local_bs, shuffle=True)
        test = DataLoader(dataset, batch_size=128)
        return train, test

    def update_weights(self, net, seed, net_glob, epoch, mu=1, lr=None):
        net_glob = net_glob

        net.train()
        # train and update
        if lr is None:
            optimizer = torch.optim.SGD(
                net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        else:
            optimizer = torch.optim.SGD(
                net.parameters(), lr=lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []
            # label_debug = [0 for i in range(100)]       ######
            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                # for label in labels:
                    # label_debug[label] += 1         #########
                labels = labels.long()
                net.zero_grad()
                logits = net(images)
                criterion = self.get_loss()
                loss = criterion(logits, labels)

                if self.args.beta > 0:
                    if batch_idx > 0:
                        w_diff = torch.tensor(0.).to(self.args.device)
                        for w, w_t in zip(net_glob.parameters(), net.parameters()):
                            w_diff += torch.pow(torch.norm(w - w_t), 2)
                        w_diff = torch.sqrt(w_diff)
                        loss += self.args.beta * mu * w_diff

                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            # print(label_debug)
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


    def update_weights_ditto(self, net, seed, net_glob, epoch, mu=0.01, lr=None):
        net_glob = net_glob

        net.train()
        # train and update
        if lr is None:
            optimizer = torch.optim.SGD(
                net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        else:
            optimizer = torch.optim.SGD(
                net.parameters(), lr=lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []
            # label_debug = [0 for i in range(100)]       ######
            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                # for label in labels:
                    # label_debug[label] += 1         #########
                labels = labels.long()
                net.zero_grad()
                logits = net(images)
                criterion = self.get_loss()
                loss = criterion(logits, labels)
                # print(loss)
                if batch_idx > 0:
                    w_diff = torch.tensor(0.).to(self.args.device)
                    for w, w_t in zip(net_glob.parameters(), net.parameters()):
                        w_diff += torch.pow(torch.norm(w - w_t), 2)
                    w_diff = torch.sqrt(w_diff)
                    loss += mu * w_diff
                # print(loss)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            # print(label_debug)
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def update_weights_backbone_only(self, net, seed, epoch, criterion=None, mu=1, lr=None):
        # 再固定head，训练表征
        count = 0
        for p in net.parameters():
            if count >= 108:        # 108
                p.requires_grad = True
            else:
                p.requires_grad = False
            count += 1

        filter(lambda p: p.requires_grad, net.parameters())
        # criterion = nn.CrossEntropyLoss()
        net.train()
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        # train and update
        if lr is None:
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, net.parameters()), lr=self.args.lr, momentum=self.args.momentum)
        else:
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=self.args.momentum)

        epoch_loss = []

        # 假设 model 是你的模型
        linear_weights = net.linear.weight


        # 创建一个掩码，原始权重为0的位置为1，其他为0
        # spar_mask = (linear_weights == 0).float()

        # 开放尾部的authority
        # 创建一个全0的掩码
        spar_mask = torch.zeros_like(linear_weights)
        # 对权重矩阵的后75行，原始权重为0的位置为1，其他为0
        # spar_mask[-75:] = (linear_weights[-75:] == 0).float()   
        spar_mask[:25] = (linear_weights[:25] == 0).float()   


        for iter in range(epoch):
            batch_loss = []
            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                labels = labels.long()
                net.zero_grad()
                logits = net(images)
                loss = criterion(logits, labels)

                loss.backward()


                # 在优化步骤之前，应用掩码
                net.linear.weight.grad *= spar_mask
                # Assume you have a model 'model'

                # After calling loss.backward(), you can check the gradients
                # max_grad = max(p.grad.data.abs().max() for p in net.parameters() if p.grad is not None)
                # print('Max gradient:', max_grad)

                optimizer.step()

                batch_loss.append(loss.item())
                # print(criterion.pn_diff[70])
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    

    def update_weights_fedrep(self, net, seed, net_glob, epoch, mu=1, lr=None):
        net_glob = net_glob

        net.train()
        # train and update
        if lr is None:
            optimizer = torch.optim.SGD(
                net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        else:
            optimizer = torch.optim.SGD(
                net.parameters(), lr=lr, momentum=self.args.momentum)

        epoch_loss = []

        # 先训练head，固定表征
        count = 0
        for p in net.parameters():
            if count >= 108:        # 108
                p.requires_grad = True
            else:
                p.requires_grad = False
            count += 1

        filter(lambda p: p.requires_grad, net.parameters())



        for iter in range(15):
            if iter == 10:
                # 再固定head，训练表征
                count = 0
                for p in net.parameters():
                    if count >= 108:        # 108
                        p.requires_grad = False
                    else:
                        p.requires_grad = True
                    count += 1

                filter(lambda p: p.requires_grad, net.parameters())

            batch_loss = []
            # label_debug = [0 for i in range(100)]       ######
            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                # for label in labels:
                    # label_debug[label] += 1         #########
                labels = labels.long()
                net.zero_grad()
                logits = net(images)
                criterion = self.get_loss()
                loss = criterion(logits, labels)

                if self.args.beta > 0:
                    if batch_idx > 0:
                        w_diff = torch.tensor(0.).to(self.args.device)
                        for w, w_t in zip(net_glob.parameters(), net.parameters()):
                            w_diff += torch.pow(torch.norm(w - w_t), 2)
                        w_diff = torch.sqrt(w_diff)
                        loss += self.args.beta * mu * w_diff

                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            # print(label_debug)
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


    def update_weights_gaux(self, net, g_head, g_aux, l_head, epoch, mu=1, lr=None, loss_switch=None):
        net.train()
        # train and update
        optimizer_g_backbone = torch.optim.SGD(list(net.parameters()), lr=self.args.lr, momentum=self.args.momentum)
        optimizer_g_aux = torch.optim.SGD(g_aux.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        # optimizer_g_aux = torch.optim.SGD([
        #                     {'params': g_aux.weight, 'weight_decay': 1e-1},  # 对权重使用weight_decay
        #                     {'params': g_aux.bias, 'weight_decay': 0}  # 对偏置不使用weight_decay
        #                 ], lr=self.args.lr, momentum=self.args.momentum)
        optimizer_l_head = torch.optim.SGD(l_head.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        # 定义优化器
        # optimizer_l_head = torch.optim.SGD([
        #                     {'params': l_head.weight, 'weight_decay': 1e-1},  # 对权重使用weight_decay
        #                     {'params': l_head.bias, 'weight_decay': 0}  # 对偏置不使用weight_decay
        #                 ], lr=self.args.lr, momentum=self.args.momentum)

        criterion_l = nn.CrossEntropyLoss()
        criterion_g = nn.CrossEntropyLoss()

        def adaptive_angle_loss(features, labels):
            similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1)
            diff_mask = labels.unsqueeze(1) != labels.unsqueeze(0)
            same_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
            loss_diff = (similarity_matrix * diff_mask.float()).sum() / diff_mask.float().sum()
            loss_same = ((1 - similarity_matrix) * same_mask.float()).sum() / same_mask.float().sum()
            return loss_diff + loss_same

        def normalized_feature_loss(features):
            # 计算特征向量的范数
            norms = torch.norm(features, dim=1)
            # 计算范数的均值
            mean_norm = torch.mean(norms)
            # 计算范数与均值的差的平方，然后求和以得到方差
            variance = torch.mean((norms - mean_norm) ** 2)
            # 返回范数的方差
            return variance

        def get_mma_loss(features, labels):
            # computing cosine similarity: dot product of normalized weight vectors
            weight_ = F.normalize(features, p=2, dim=1)
            cosine = torch.matmul(weight_, weight_.t())  
            same_mask = labels.unsqueeze(1) == labels.unsqueeze(0)   

            # make sure that the diagnonal elements cannot be selected
            cosine = cosine - 2. * torch.diag(torch.diag(cosine))  
            cosine[same_mask] = -1

            # maxmize the minimum angle
            loss = -torch.acos(cosine.max(dim=1)[0].clamp(-0.99999, 0.99999)).mean()

            return loss

        
        def balanced_softmax_loss(labels, logits, sample_per_class=None):
            """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
            Args:
            labels: A int tensor of size [batch].
            logits: A float tensor of size [batch, no_of_classes].
            sample_per_class: A int tensor of size [no of classes].
            reduction: string. One of "none", "mean", "sum"
            Returns:
            loss: A float tensor. Balanced Softmax Loss.
            """
            if sample_per_class is None:
                sample_per_class = [500, 477, 455, 434, 415, 396, 378, 361, 344, 328, 314, 299, 286, 273, 260, 248, 237, 226, 216, 206, 197, 188, 179, 171, 163, 156, 149, 142, 135, 129, 123, 118, 112, 107, 102, 98, 93, 89, 85, 81, 77, 74, 70, 67, 64, 61, 58, 56, 53, 51, 48, 46, 44, 42, 40, 38, 36, 35, 33, 32, 30, 29, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 15, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8, 7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5]
            sample_per_class = torch.tensor(sample_per_class)
            spc = sample_per_class.type_as(logits)
            spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
            logits = logits + spc.log()
            loss = F.cross_entropy(input=logits, target=labels, reduction='mean')
            return loss

        
        if loss_switch == "focus_loss":
            criterion_l = focus_loss(num_classes=100)

        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []
            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)

                labels = labels.long()
                optimizer_g_backbone.zero_grad()
                optimizer_g_aux.zero_grad()
                optimizer_l_head.zero_grad()
                # net.zero_grad()

                # backbone
                features = net(images, latent_output=True)

                # mma_loss = get_mma_loss(features, labels)


                output_g_backbone = g_head(features)
            

                
                loss_g_backbone = criterion_g(output_g_backbone, labels)
                # dist_est = torch.pow(torch.norm(g_aux.weight, p=2, dim=1), 3)
                # loss_g_backbone = balanced_softmax_loss(labels, output_g_backbone, sample_per_class = dist_est)
                loss_g_backbone.backward()
                # max_grad = max(p.grad.data.abs().max() for p in net.parameters() if p.grad is not None)
                # print('Max gradient:', max_grad)
                optimizer_g_backbone.step()
                
                # g_aux
                output_g_aux = g_aux(features.detach())
                loss_g_aux = criterion_l(output_g_aux, labels)
                loss_g_aux.backward()
                optimizer_g_aux.step()

                # p_head
                output_l_head = l_head(features.detach())
                loss_l_head = criterion_l(output_l_head, labels)
                loss_l_head.backward()
                optimizer_l_head.step()

                loss = loss_g_backbone + loss_g_aux + loss_l_head
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), g_aux, l_head, sum(epoch_loss) / len(epoch_loss)





    def update_weights_class_mean(self, net, g_head, g_aux, l_head, epoch, class_means, mu=1, lr=None, loss_switch=None):
        net.train()
        # train and update
        optimizer_g_backbone = torch.optim.SGD(list(net.parameters()), lr=self.args.lr, momentum=self.args.momentum)
        optimizer_g_aux = torch.optim.SGD(g_aux.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        # optimizer_g_aux = torch.optim.SGD([
        #                     {'params': g_aux.weight, 'weight_decay': 1e-1},  # 对权重使用weight_decay
        #                     {'params': g_aux.bias, 'weight_decay': 0}  # 对偏置不使用weight_decay
        #                 ], lr=self.args.lr, momentum=self.args.momentum)
        optimizer_l_head = torch.optim.SGD(l_head.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        # 定义优化器
        # optimizer_l_head = torch.optim.SGD([
        #                     {'params': l_head.weight, 'weight_decay': 1e-1},  # 对权重使用weight_decay
        #                     {'params': l_head.bias, 'weight_decay': 0}  # 对偏置不使用weight_decay
        #                 ], lr=self.args.lr, momentum=self.args.momentum)

        criterion_l = nn.CrossEntropyLoss()
        criterion_g = nn.CrossEntropyLoss()

        def inner_min(features, labels, class_means):
            valid_class_means = [class_mean for class_mean in class_means.values() if class_mean is not None]
            if valid_class_means:
                global_mean = torch.stack(valid_class_means).mean(dim=0)
            del valid_class_means
            # output: global_mean
            # 归一化之后的
            class_means = {class_idx: (class_mean - global_mean) if class_mean is not None else None for class_idx, class_mean in class_means.items()}
            features = features - global_mean

            loss = 0
            for i in range(len(features)):
                loss += F.cosine_similarity(features[i].unsqueeze(0), class_means[labels[i].item()].unsqueeze(0))
            return loss


        def inter_max(features, labels, class_means):
            valid_class_means = [class_mean for class_mean in class_means.values() if class_mean is not None]
            if valid_class_means:
                global_mean = torch.stack(valid_class_means).mean(dim=0)
            del valid_class_means
            class_means = {class_idx: (class_mean - global_mean) if class_mean is not None else None for class_idx, class_mean in class_means.items()}
            features = features - global_mean

            # 1.统计有哪些label 2.看这些label的class mean距离哪个class mean比较大 3.计算features与这个class mean的相似度，最小化这个相似度
            # 仅选择不为None的class_means
            filtered_class_means = {k: v for k, v in class_means.items() if v is not None}
            all_class_means = torch.stack(list(filtered_class_means.values()))
            # 初始化用于存储每个feature与其"最相似的class mean"相似度的张量
            max_similarities = torch.zeros(len(features), device=features.device)


            for i in range(len(features)):
                current_label = labels[i].item()
                current_class_mean = class_means[current_label]

                # 找出除当前class_mean外的所有其他class_mean的索引
                other_class_mean_indices = [idx for idx, k in enumerate(filtered_class_means.keys()) if k != current_label]

                # 计算当前class_mean与所有其他class_means的相似度
                similarities = F.cosine_similarity(current_class_mean.unsqueeze(0), all_class_means[other_class_mean_indices], dim=1)

                # 找到最相似的class_mean的相似度和索引
                max_similarity, max_id = similarities.max(dim=0)

                # 使用原始索引来找出最相似的class_mean
                most_similar_class_mean = all_class_means[other_class_mean_indices[max_id]]
                
                # 计算该feature与这个最相似的class_mean的相似度
                feature_similarity = F.cosine_similarity(features[i].unsqueeze(0), most_similar_class_mean.unsqueeze(0), dim=1)

                # 存储相似度
                max_similarities[i] = feature_similarity

            # 最终的损失是所有max_similarities的和
            total_loss = max_similarities.sum()
            return total_loss



        def inter_max_feat_classmean(features, labels, class_means):
            # 初始化用于存储每个feature与其"最相似的class mean"相似度的张量
            max_similarities = torch.zeros(len(features), device=features.device)

            # 预处理：去除None值，计算global_mean
            valid_class_means = [class_mean for class_mean in class_means.values() if class_mean is not None]
            if valid_class_means:
                global_mean = torch.stack(valid_class_means).mean(dim=0)
            del valid_class_means

            # 去全局均值化
            features = features - global_mean
            class_means = {class_idx: (class_mean - global_mean) if class_mean is not None else None for class_idx, class_mean in class_means.items()}

            # 仅选择不为None的class_means
            filtered_class_means = {k: v for k, v in class_means.items() if v is not None}
            all_class_means = torch.stack(list(filtered_class_means.values()))

            for i in range(len(features)):
                current_label = labels[i].item()

                # 找出除当前class_mean外的所有其他class_mean的索引
                other_class_mean_indices = [idx for idx, k in enumerate(filtered_class_means.keys()) if k != current_label]

                # 直接使用features与所有其他的class_means来计算相似度
                similarities = F.cosine_similarity(features[i].unsqueeze(0), all_class_means[other_class_mean_indices], dim=1)

                # 找到最相似的class_mean的相似度和索引
                max_similarity, max_id = similarities.max(dim=0)

                # 使用原始索引来找出最相似的class_mean
                most_similar_class_mean = all_class_means[other_class_mean_indices[max_id]]

                # 计算该feature与这个最相似的class_mean的相似度
                feature_similarity = F.cosine_similarity(features[i].unsqueeze(0), most_similar_class_mean.unsqueeze(0), dim=1)

                # 存储相似度
                max_similarities[i] = feature_similarity

            # 最终的损失是所有max_similarities的和
            total_loss = max_similarities.sum()
            return total_loss


        def balanced_softmax_loss(labels, logits, sample_per_class=None):
            """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
            Args:
            labels: A int tensor of size [batch].
            logits: A float tensor of size [batch, no_of_classes].
            sample_per_class: A int tensor of size [no of classes].
            reduction: string. One of "none", "mean", "sum"
            Returns:
            loss: A float tensor. Balanced Softmax Loss.
            """
            if sample_per_class is None:
                sample_per_class = [500, 477, 455, 434, 415, 396, 378, 361, 344, 328, 314, 299, 286, 273, 260, 248, 237, 226, 216, 206, 197, 188, 179, 171, 163, 156, 149, 142, 135, 129, 123, 118, 112, 107, 102, 98, 93, 89, 85, 81, 77, 74, 70, 67, 64, 61, 58, 56, 53, 51, 48, 46, 44, 42, 40, 38, 36, 35, 33, 32, 30, 29, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 15, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8, 7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5]
            sample_per_class = torch.tensor(sample_per_class)
            spc = sample_per_class.type_as(logits)
            spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
            logits = logits + spc.log()
            loss = F.cross_entropy(input=logits, target=labels, reduction='mean')
            return loss

        
        if loss_switch == "focus_loss":
            criterion_l = focus_loss(num_classes=100)

        epoch_loss = []
        momentum = 0.9  # 可以根据实际情况调整
        for iter in range(epoch):
            batch_loss = []
            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)

                labels = labels.long()
                optimizer_g_backbone.zero_grad()
                optimizer_g_aux.zero_grad()
                optimizer_l_head.zero_grad()
                # net.zero_grad()

                # backbone
                features = net(images, latent_output=True)
                
                # 更新 class means 使用指数移动平均
                for i in range(len(features)):
                    class_idx = labels[i].item()

                    if class_means[class_idx] is None:
                        class_means[class_idx] = features[i]
                    else:
                        class_means[class_idx] = momentum * class_means[class_idx] + (1 - momentum) * features[i]
                for key in class_means:
                    if class_means[key] != None:
                        class_means[key] = class_means[key].detach()

                # inner_loss = inner_min(features, labels, copy.deepcopy(class_means))
                # inter_loss = inter_max_feat_classmean(features, labels, copy.deepcopy(class_means))

                # mma_loss = get_mma_loss(class_means, features)

                output_g_backbone = g_head(features)
            
                # mma_loss = get_mma_loss(output_g_backbone, labels)
                # dist_est = torch.pow(torch.norm(g_aux.weight, p=2, dim=1), 3)

                # loss_g_backbone = balanced_softmax_loss(labels, output_g_backbone, sample_per_class=torch.pow(torch.norm(g_aux.weight, p=2, dim=1), 3)) + 0.1 * inter_loss

                loss_g_backbone = balanced_softmax_loss(labels, output_g_backbone) 
                loss_g_backbone.backward()


                # max_grad = max(p.grad.data.abs().max() for p in net.parameters() if p.grad is not None)
                # print('Max gradient:', max_grad)
                optimizer_g_backbone.step()
                
                # g_aux
                output_g_aux = g_aux(features.detach())
                loss_g_aux = criterion_l(output_g_aux, labels)
                loss_g_aux.backward()
                optimizer_g_aux.step()

                # p_head
                output_l_head = l_head(features.detach())
                loss_l_head = criterion_l(output_l_head, labels)
                loss_l_head.backward()
                optimizer_l_head.step()

                loss = loss_g_backbone + loss_g_aux + loss_l_head
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), g_aux, l_head, sum(epoch_loss) / len(epoch_loss), class_means
    

    def update_weights_etf(self, net, g_head, g_aux, l_head, epoch, mu=1, lr=None, loss_switch=None):
        net.train()
        # train and update
        optimizer_g_backbone = torch.optim.SGD([{"params": net.parameters()},
                                {"params": g_head.parameters()}], lr=self.args.lr, momentum=self.args.momentum)
        optimizer_g_aux = torch.optim.SGD(g_aux.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        # optimizer_g_aux = torch.optim.SGD([
        #                     {'params': g_aux.weight, 'weight_decay': 1e-1},  # 对权重使用weight_decay
        #                     {'params': g_aux.bias, 'weight_decay': 0}  # 对偏置不使用weight_decay
        #                 ], lr=self.args.lr, momentum=self.args.momentum)
        optimizer_l_head = torch.optim.SGD(l_head.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        # 定义优化器
        # optimizer_l_head = torch.optim.SGD([
        #                     {'params': l_head.weight, 'weight_decay': 1e-1},  # 对权重使用weight_decay
        #                     {'params': l_head.bias, 'weight_decay': 0}  # 对偏置不使用weight_decay
        #                 ], lr=self.args.lr, momentum=self.args.momentum)


        criterion_l = nn.CrossEntropyLoss()
            
        num_classes = 100
        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []
            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)

                labels = labels.long()
                optimizer_g_backbone.zero_grad()
                optimizer_g_aux.zero_grad()
                optimizer_l_head.zero_grad()
                # net.zero_grad()1


                # 更新ori_M(ETF权重)
                learned_norm = produce_Ew(labels, num_classes)
                cur_M = learned_norm * g_head.ori_M


                # backbone
                feat = net(images, latent_output=True)
                features = g_head(feat)

                output = torch.matmul(features, cur_M)
                with torch.no_grad():
                    feat_nograd = features.detach()
                    H_length = torch.clamp(torch.sqrt(torch.sum(feat_nograd ** 2, dim=1, keepdims=False)), 1e-8)
                loss_g_backbone = dot_loss(features, labels, cur_M, g_head, 'reg_dot_loss', H_length, reg_lam=0)
                # loss_g_backbone = criterion_g(output_g_backbone, labels)
                loss_g_backbone.backward()
                # max_grad = max(p.grad.data.abs().max() for p in net.parameters() if p.grad is not None)
                # print('Max gradient:', max_grad)
                optimizer_g_backbone.step()
                
                # g_aux
                output_g_aux = g_aux(feat.detach())
                loss_g_aux = criterion_l(output_g_aux, labels)
                loss_g_aux.backward()
                optimizer_g_aux.step()

                # p_head
                output_l_head = l_head(feat.detach())
                loss_l_head = criterion_l(output_l_head, labels)
                loss_l_head.backward()
                optimizer_l_head.step()

                loss = loss_g_backbone + loss_g_aux + loss_l_head
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), g_aux, l_head, sum(epoch_loss) / len(epoch_loss)
    

    def update_weights_auto_selective_ghead(self, net, g_head, g_aux, l_head, epoch, mu=1, lr=None, loss_switch=None):
        net.train()
        # train and update
        optimizer_g_backbone = torch.optim.SGD(list(net.parameters()) + [g_head.weights], lr=self.args.lr, momentum=self.args.momentum)
        optimizer_g_aux = torch.optim.SGD(g_aux.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        # optimizer_g_aux = torch.optim.SGD([
        #                     {'params': g_aux.weight, 'weight_decay': 1e-1},  # 对权重使用weight_decay
        #                     {'params': g_aux.bias, 'weight_decay': 0}  # 对偏置不使用weight_decay
        #                 ], lr=self.args.lr, momentum=self.args.momentum)
        optimizer_l_head = torch.optim.SGD(l_head.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        # 定义优化器
        # optimizer_l_head = torch.optim.SGD([
        #                     {'params': l_head.weight, 'weight_decay': 1e-1},  # 对权重使用weight_decay
        #                     {'params': l_head.bias, 'weight_decay': 0}  # 对偏置不使用weight_decay
        #                 ], lr=self.args.lr, momentum=self.args.momentum)

        criterion_l = nn.CrossEntropyLoss()
        criterion_g = nn.CrossEntropyLoss()
        if loss_switch == "focus_loss":
            criterion_l = focus_loss(num_classes=100)

        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []
            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)

                labels = labels.long()
                optimizer_g_backbone.zero_grad()
                optimizer_g_aux.zero_grad()
                optimizer_l_head.zero_grad()
                # net.zero_grad()

                # backbone
                features = net(images, latent_output=True)
                output_g_backbone = g_head(features)
                loss_g_backbone = criterion_g(output_g_backbone, labels)
                loss_g_backbone.backward()
                # max_grad = max(p.grad.data.abs().max() for p in net.parameters() if p.grad is not None)
                # print('Max gradient:', max_grad)
                optimizer_g_backbone.step()
                
                # g_aux
                output_g_aux = g_aux(features.detach())
                loss_g_aux = criterion_l(output_g_aux, labels)
                loss_g_aux.backward()
                optimizer_g_aux.step()

                # p_head
                output_l_head = l_head(features.detach())
                loss_l_head = criterion_l(output_l_head, labels)
                loss_l_head.backward()
                optimizer_l_head.step()

                loss = loss_g_backbone + loss_g_aux + loss_l_head
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), g_aux, g_head, l_head, sum(epoch_loss) / len(epoch_loss)
    
    def update_weights_fedrod(self, net, g_head, g_aux, l_head, epoch, mu=1, lr=None):
        net.train()
        # train and update
        # optimizer_g_backbone = torch.optim.SGD(list(net.parameters()), lr=self.args.lr, momentum=self.args.momentum)
        optimizer_g_aux = torch.optim.SGD(list(net.parameters()) + list(g_aux.parameters()), lr=self.args.lr, momentum=self.args.momentum)
        # optimizer_g_aux = torch.optim.SGD([
        #                     {'params': g_aux.weight, 'weight_decay': 1e-1},  # 对权重使用weight_decay
        #                     {'params': g_aux.bias, 'weight_decay': 0}  # 对偏置不使用weight_decay
        #                 ], lr=self.args.lr, momentum=self.args.momentum)
        optimizer_l_head = torch.optim.SGD(l_head.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        # 定义优化器
        # optimizer_l_head = torch.optim.SGD([
        #                     {'params': l_head.weight, 'weight_decay': 1e-1},  # 对权重使用weight_decay
        #                     {'params': l_head.bias, 'weight_decay': 0}  # 对偏置不使用weight_decay
        #                 ], lr=self.args.lr, momentum=self.args.momentum)

        criterion_l = nn.CrossEntropyLoss()
        criterion_g = nn.CrossEntropyLoss()

        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []
            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)

                labels = labels.long()
                optimizer_g_aux.zero_grad()
                optimizer_l_head.zero_grad()

                # backbone + g_aux
                features = net(images, latent_output=True)
                output = g_aux(features)
                loss_g = criterion_g(output, labels)
                loss_g.backward()
                optimizer_g_aux.step()

                # p_head
                output_l_head = l_head(features.detach())
                loss_l_head = criterion_l(output_l_head, labels)
                loss_l_head.backward()
                optimizer_l_head.step()

                loss = loss_g + loss_l_head
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), g_aux, l_head, sum(epoch_loss) / len(epoch_loss)
    
    def update_weights_balsoft_backup(self, net, g_head, g_aux, l_head, seed, net_glob, epoch, mu=1, lr=None):

        net.train()
        # train and update
        optimizer_g = torch.optim.SGD(list(net.parameters()), lr=self.args.lr, momentum=self.args.momentum)
        optimizer_l = torch.optim.SGD(l_head.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        optimizer_g_aux = torch.optim.SGD(g_aux.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        criterion_l = nn.CrossEntropyLoss()
        criterion_g = nn.CrossEntropyLoss()

        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []
            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)

                labels = labels.long()
                optimizer_g.zero_grad()
                optimizer_l.zero_grad()
                optimizer_g_aux.zero_grad()

                # backbone更新
                features = net(images, latent_output=True)
                output_g = g_head(features)
                loss_g = criterion_g(output_g, labels)
                loss_g.backward()
                optimizer_g.step()

                # aux更新
                output_g_aux = g_aux(features.detach())
                loss_g_aux = criterion_g(output_g_aux, labels)
                loss_g_aux.backward()
                optimizer_g_aux.step()

                # p cls更新
                output_l = l_head(features.detach())
                loss_l = criterion_l(output_l, labels)
                loss_l.backward()
                optimizer_l.step()

                loss = loss_g + loss_g_aux + loss_l
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), g_aux, l_head, sum(epoch_loss) / len(epoch_loss)
    

    def update_weights_phead_backup(self, net, g_aux, g_head, l_head, seed, net_glob, epoch, mu=1, lr=None):

        net.train()
        # train and update
        optimizer_g = torch.optim.SGD(list(net.parameters()), lr=self.args.lr, momentum=self.args.momentum)
        optimizer_l = torch.optim.SGD(l_head.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        optimizer_g_aux = torch.optim.SGD(g_aux.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        # criterion_g = balsoft_loss
        criterion_l = nn.CrossEntropyLoss()
        criterion_g = nn.CrossEntropyLoss()

        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []
            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)

                labels = labels.long()
                optimizer_g.zero_grad()
                optimizer_l.zero_grad()
                optimizer_g_aux.zero_grad()

                # backbone更新
                features = net(images, latent_output=True)
                # output_g = g_head(features)
                # loss_g = criterion_g(output_g, labels)
                # loss_g.backward()
                # optimizer_g.step()

                # aux更新
                # output_g_aux = g_aux(features.detach())
                # loss_g_aux = criterion_g(output_g_aux, labels)
                # loss_g_aux.backward()
                # optimizer_g_aux.step()

                # p cls更新
                output_l = l_head(features.detach())
                loss_l = criterion_l(output_l, labels)
                loss_l.backward()
                optimizer_l.step()

                loss = loss_l
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), g_aux, l_head, sum(epoch_loss) / len(epoch_loss)
    

    def update_weights_unlearning(self, net, g_aux, g_head, l_head, seed, net_glob, epoch, mu=1, lr=None):

        net.train()
        # train and update
        optimizer_g = torch.optim.SGD(list(net.parameters()), lr=self.args.lr, momentum=self.args.momentum)
        optimizer_l = torch.optim.SGD(list(l_head.parameters()), lr=self.args.lr, momentum=self.args.momentum)
        optimizer_g_aux = torch.optim.SGD(list(g_aux.parameters()), lr=self.args.lr, momentum=self.args.momentum)
        optimizer = torch.optim.SGD(list(g_aux.parameters()) + list(l_head.parameters()), lr=self.args.lr, momentum=self.args.momentum)
        # criterion_g = balsoft_loss
        criterion_l = nn.CrossEntropyLoss()
        criterion_g = nn.CrossEntropyLoss()
        
        criterion_kl = nn.KLDivLoss(reduction='batchmean')
        criterion_ce = nn.CrossEntropyLoss()

        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []
            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)

                labels = labels.long()
                # optimizer_g.zero_grad()
                # optimizer_l.zero_grad()
                optimizer_g_aux.zero_grad()
                # optimizer.zero_grad()

                # backbone更新
                features = net(images, latent_output=True)
                # output_g = g_head(features)
                # loss_g = criterion_g(output_g, labels)
                # loss_g.backward()
                # optimizer_g.step()

                # aux更新
                # output_g_aux = g_aux(features.detach())
                # loss_g_aux = criterion_g(output_g_aux, labels)
                # loss_g_aux.backward()
                # optimizer_g_aux.step()

                # 迁移学习
                outputs_g_aux = g_aux(features.detach())
                outputs_l_head = l_head(features.detach())
                
                outputs_g_aux_normalized = F.normalize(outputs_g_aux, dim=1)
                outputs_l_head_normalized = F.normalize(outputs_l_head, dim=1)

                
                # loss_ce = criterion_ce(outputs_g_aux, labels)
                loss_kl = criterion_kl(nn.functional.log_softmax(outputs_g_aux_normalized, dim=1),
                                    nn.functional.softmax(outputs_l_head_normalized, dim=1))
                loss = loss_kl
                loss.backward()
                optimizer_g_aux.step()


                # p cls更新
                # output_l = l_head(features.detach())
                # loss_l = criterion_l(output_l, labels)
                # loss_l.backward()
                # optimizer_l.step()

                # loss = loss_l
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), copy.deepcopy(g_aux), copy.deepcopy(l_head), sum(epoch_loss) / len(epoch_loss)
    
## gain personalzied l_head
    def update_weights_norm_init(self, net, g_aux, g_head, l_head, seed, net_glob, epoch, mu=1, lr=None):

        net.train()
        # train and update


        # 权重norm初始化
        norm = torch.norm(l_head.weight, p=2, dim=1)
        # 将g_head.weight转换为torch.nn.Parameter类型
        g_aux.weight = nn.Parameter(g_aux.weight * norm.unsqueeze(1))


        optimizer_g = torch.optim.SGD(list(net.parameters()), lr=self.args.lr, momentum=self.args.momentum)
        optimizer_l = torch.optim.SGD(l_head.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        optimizer_g_aux = torch.optim.SGD(g_aux.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        optimizer = torch.optim.SGD(list(g_aux.parameters()) + list(l_head.parameters()), lr=self.args.lr, momentum=self.args.momentum)
        # criterion_g = balsoft_loss
        criterion_l = nn.CrossEntropyLoss()
        criterion_g = nn.CrossEntropyLoss()
        
        # criterion_kl = nn.KLDivLoss(reduction='batchmean')
        criterion_ce = nn.CrossEntropyLoss()

        epoch_loss = []


        for iter in range(epoch):
            batch_loss = []
            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)

                labels = labels.long()
                # optimizer_g.zero_grad()
                # optimizer_l.zero_grad()
                optimizer_g_aux.zero_grad()
                # optimizer.zero_grad()

                # backbone更新
                features = net(images, latent_output=True)
                # output_g = g_head(features)
                # loss_g = criterion_g(output_g, labels)
                # loss_g.backward()
                # optimizer_g.step()

                # aux更新
                # output_g_aux = g_aux(features.detach())
                # loss_g_aux = criterion_g(output_g_aux, labels)
                # loss_g_aux.backward()
                # optimizer_g_aux.step()

                # 迁移学习
                outputs_g_aux = g_aux(features.detach())
                # outputs_l_head = l_head(features.detach())
                
                # outputs_g_aux_normalized = F.normalize(outputs_g_aux, dim=1)
                # outputs_l_head_normalized = F.normalize(outputs_l_head, dim=1)

                
                loss = criterion_ce(outputs_g_aux, labels)
                # loss_kl = criterion_kl(nn.functional.log_softmax(outputs_l_head_normalized, dim=1),
                #                     nn.functional.softmax(outputs_g_aux_normalized, dim=1))
                # loss = loss_kl
                loss.backward()
                optimizer_g_aux.step()
                


                # p cls更新
                # output_l = l_head(features.detach())
                # loss_l = criterion_l(output_l, labels)
                # loss_l.backward()
                # optimizer_l.step()

                # loss = loss_l
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), copy.deepcopy(g_aux), copy.deepcopy(l_head), sum(epoch_loss) / (len(epoch_loss) + 1e-10)
    
    def pfedme_update_weights(self, net, seed, net_glob, epoch, mu=1, lr=None):
        net_glob = net_glob

        net.train()
        # train and update
        if lr is None:
            optimizer = torch.optim.SGD(
                net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        else:
            optimizer = torch.optim.SGD(
                net.parameters(), lr=lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []
            # label_debug = [0 for i in range(100)]       ######
            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                # for label in labels:
                    # label_debug[label] += 1         #########
                labels = labels.long()
                net.zero_grad()
                logits = net(images)
                criterion = self.get_loss()
                loss = criterion(logits, labels)

                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            # print(label_debug)
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    # 专为获取梯度信息设计的local update函数
    def update_weights_gasp_grad(self, net, seed, net_glob, client_id, epoch, gradBag, mu=1, lr=None):
        hookObj,  gradAnalysor = gradBag.get_client(client_id)
        net_glob = net_glob
        net.train()
        # train and update
        if lr is None:
            optimizer = torch.optim.SGD(
                net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        else:
            optimizer = torch.optim.SGD(
                net.parameters(), lr=lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []
            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)

                labels = labels.long()
                net.zero_grad()
                logits = net(images)
                criterion = self.get_loss()
                loss = criterion(logits, labels)
                hook_handle = logits.register_hook(
                    hookObj.hook_func_tensor)  # hook抓取梯度
                if self.args.beta > 0:
                    if batch_idx > 0:
                        w_diff = torch.tensor(0.).to(self.args.device)
                        for w, w_t in zip(net_glob.parameters(), net.parameters()):
                            w_diff += torch.pow(torch.norm(w - w_t), 2)
                        w_diff = torch.sqrt(w_diff)
                        loss += self.args.beta * mu * w_diff
                loss.backward()

                if hookObj.has_gradient():
                    # 输入一个batch的梯度和label
                    gradAnalysor.update(
                        hookObj.get_gradient(), labels.cpu().numpy().tolist())
                optimizer.step()
                hook_handle.remove()
                batch_loss.append(loss.item())
                gradAnalysor.print_for_debug()

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        gradBag.load_in(client_id, hookObj, gradAnalysor)
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), gradBag
# 用pid算法进行local update

    def update_weights_GBA_Loss(self, net, seed, epoch, pidloss, mu=1, lr=None):
        
        net.train()
        # train and update
        if lr is None:
            optimizer = torch.optim.SGD(
                net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        else:
            optimizer = torch.optim.SGD(
                net.parameters(), lr=lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []
            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)

                labels = labels.long()
                net.zero_grad()
                logits = net(images)
                criterion = pidloss
                loss = criterion(logits, labels)

                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


    def update_weights_GBA_Finetune(self, net, seed, epoch, pidloss, mu=1, lr=None):
        # 冻结一部分层
        count = 0
        for p in net.parameters():
            if count >= 105:        # 108
                break
            p.requires_grad = False
            count += 1

        filter(lambda p: p.requires_grad, net.parameters())
        net.train()
        # train and update
        if lr is None:
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, net.parameters()), lr=self.args.lr, momentum=self.args.momentum)
        else:
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []
            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)

                labels = labels.long()
                net.zero_grad()
                logits = net(images)
                criterion = pidloss
                loss = criterion(logits, labels)

                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
                # print(criterion.pn_diff[70])
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    # 用pid算法进行local update
    def update_weights_GBA_Layer(self, net, seed, epoch, GBA_Loss, GBA_Layer, mu=1, lr=None):

        net.train()
        GBA_Layer.train()
        # train and update
        if lr is None:
            backbone_optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        else:
            backbone_optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []
            # label_sum = [0 for i in range(10)]
            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                labels = labels.long()
                # for i in labels:
                #     label_sum[i] += 1

                # init
                net.zero_grad()
                feat = net(images)
                logits = GBA_Layer(feat)
                loss = GBA_Loss(logits, labels) 
                loss.backward()
                backbone_optimizer.step()
                
                # print(GBA_Loss.pn_diff[52])
                # loss
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        # print("-----------------------------------------------------------------------")
        return net.state_dict(), GBA_Layer.state_dict(), sum(epoch_loss) / len(epoch_loss)
# global dataset is balanced
def globaltest_villina(net, test_dataset, args, dataset_class=None):
    global_test_distribution = dataset_class.global_test_distribution
    three_shot_dict, _ = shot_split(global_test_distribution, threshold_3shot=[75, 95])
    correct_3shot = {"head": 0, "middle": 0, "tail": 0}
    total_3shot = {"head": 0, "middle": 0, "tail": 0}
    acc_3shot_global = {"head": None, "middle": None, "tail": None}
    net.eval()
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=100, shuffle=False)
    # 监视真实情况下所有样本的类别分布
    total_class_label = [0 for i in range(args.num_classes)]

    # global_test_distribution = global_test_distribution + 100*[0]
    # zero_classes = np.where(global_test_distribution == 0)[0]
    # for i in zero_classes:
    #     net.linear.weight.data[i, :] = -1e10

    predict_true_class = [0 for i in range(args.num_classes)]
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # class-wise acc calc
            for i in range(len(labels)):
                total_class_label[int(labels[i])] += 1      # total
                if predicted[i] == labels[i]:
                    predict_true_class[int(labels[i])] += 1

            # start: cal 3shot metrics
            for label in labels:
                if label in three_shot_dict["head"]:
                    total_3shot["head"] += 1
                elif label in three_shot_dict["middle"]:
                    total_3shot["middle"] += 1
                else:
                    total_3shot["tail"] += 1
            for i in range(len(predicted)):
                if predicted[i] == labels[i] and labels[i] in three_shot_dict["head"]:   # 预测正确且在head中
                    correct_3shot["head"] += 1
                # 预测正确且在middle中
                elif predicted[i] == labels[i] and labels[i] in three_shot_dict["middle"]:
                    correct_3shot["middle"] += 1
                # 预测正确且在tail中
                elif predicted[i] == labels[i] and labels[i] in three_shot_dict["tail"]:
                    correct_3shot["tail"] += 1      # 在tail中
            # end
    acc_class_wise = [predict_true_class[i] / (total_class_label[i] + 1e-10) for i in range(args.num_classes)]
    # print(acc_class_wise)
    acc = correct / total
    acc_3shot_global["head"] = correct_3shot["head"] / \
        (total_3shot["head"] + 1e-10)
    acc_3shot_global["middle"] = correct_3shot["middle"] / \
        (total_3shot["middle"] + 1e-10)
    acc_3shot_global["tail"] = correct_3shot["tail"] / \
        (total_3shot["tail"] + 1e-10)
    return acc, acc_3shot_global


def globaltest(net, g_head, test_dataset, args, dataset_class=None, head_switch=True):
    global_test_distribution = dataset_class.global_test_distribution
    three_shot_dict, _ = shot_split(global_test_distribution, threshold_3shot=[75, 95])
    correct_3shot = {"head": 0, "middle": 0, "tail": 0}
    total_3shot = {"head": 0, "middle": 0, "tail": 0}
    acc_3shot_global = {"head": None, "middle": None, "tail": None}
    net.eval()
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=100, shuffle=False)
    # 监视真实情况下所有样本的类别分布
    total_class_label = [0 for i in range(args.num_classes)]
    predict_true_class = [0 for i in range(args.num_classes)]
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            features = net(images, latent_output=True)
            if head_switch == True:
                outputs = g_head(features)
            else:
                outputs = features
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # class-wise acc calc
            for i in range(len(labels)):
                total_class_label[int(labels[i])] += 1      # total
                if predicted[i] == labels[i]:
                    predict_true_class[int(labels[i])] += 1

            # start: cal 3shot metrics
            for label in labels:
                if label in three_shot_dict["head"]:
                    total_3shot["head"] += 1
                elif label in three_shot_dict["middle"]:
                    total_3shot["middle"] += 1
                else:
                    total_3shot["tail"] += 1
            for i in range(len(predicted)):
                if predicted[i] == labels[i] and labels[i] in three_shot_dict["head"]:   # 预测正确且在head中
                    correct_3shot["head"] += 1
                # 预测正确且在middle中
                elif predicted[i] == labels[i] and labels[i] in three_shot_dict["middle"]:
                    correct_3shot["middle"] += 1
                # 预测正确且在tail中
                elif predicted[i] == labels[i] and labels[i] in three_shot_dict["tail"]:
                    correct_3shot["tail"] += 1      # 在tail中
            # end
    acc_class_wise = [predict_true_class[i] / total_class_label[i] for i in range(args.num_classes)]
    acc = correct / total
    acc_3shot_global["head"] = correct_3shot["head"] / \
        (total_3shot["head"] + 1e-10)
    acc_3shot_global["middle"] = correct_3shot["middle"] / \
        (total_3shot["middle"] + 1e-10)
    acc_3shot_global["tail"] = correct_3shot["tail"] / \
        (total_3shot["tail"] + 1e-10)
    return acc, acc_3shot_global


def globaltest_calibra(net, g_aux, test_dataset, args, dataset_class=None, head_switch=True):
    global_test_distribution = dataset_class.global_test_distribution
    three_shot_dict, _ = shot_split(global_test_distribution, threshold_3shot=[75, 95])
    correct_3shot = {"head": 0, "middle": 0, "tail": 0}
    total_3shot = {"head": 0, "middle": 0, "tail": 0}
    acc_3shot_global = {"head": None, "middle": None, "tail": None}
    net.eval()
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=100, shuffle=False)
    # 监视真实情况下所有样本的类别分布
    total_class_label = [0 for i in range(args.num_classes)]
    predict_true_class = [0 for i in range(args.num_classes)]
    cali_alpha = torch.norm(g_aux.weight, dim=1)


    # 矫正feats
    # 计算 cali_alpha 的倒数
    cali_alpha = torch.pow(cali_alpha, 1)
    inverse_cali_alpha = 1.7 / cali_alpha
    # 将 inverse_cali_alpha 扩展为 (100, 1) 的形状
    inverse_cali_alpha = inverse_cali_alpha.view(-1, 1)
    

    # 矫正cls
    g_aux.weight = torch.nn.Parameter(g_aux.weight * inverse_cali_alpha)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            features = net(images, latent_output=True)
            # 利用广播机制，将 features 的每个元素乘以 inverse_cali_alpha
            # features = features * inverse_cali_alpha
            if head_switch == True:
                outputs = g_aux(features)
                # outputs = features.matmul(g_head.weight.t()) + g_head.bias
                # 计算 features 的 Frobenius 范数
                # features_norm = torch.norm(features, dim=1, keepdim=True)

                # # 计算 g_head.weight 的 Frobenius 范数
                # weight_norm = torch.norm(g_head.weight, dim=1, keepdim=True)

                # # 计算两个范数的乘积，并转置得到 [100, 100] 的输出
                # outputs = (features_norm * weight_norm.t()).squeeze()
            else:
                outputs = features
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # class-wise acc calc
            for i in range(len(labels)):
                total_class_label[int(labels[i])] += 1      # total
                if predicted[i] == labels[i]:
                    predict_true_class[int(labels[i])] += 1

            # start: cal 3shot metrics
            for label in labels:
                if label in three_shot_dict["head"]:
                    total_3shot["head"] += 1
                elif label in three_shot_dict["middle"]:
                    total_3shot["middle"] += 1
                else:
                    total_3shot["tail"] += 1
            for i in range(len(predicted)):
                if predicted[i] == labels[i] and labels[i] in three_shot_dict["head"]:   # 预测正确且在head中
                    correct_3shot["head"] += 1
                # 预测正确且在middle中
                elif predicted[i] == labels[i] and labels[i] in three_shot_dict["middle"]:
                    correct_3shot["middle"] += 1
                # 预测正确且在tail中
                elif predicted[i] == labels[i] and labels[i] in three_shot_dict["tail"]:
                    correct_3shot["tail"] += 1      # 在tail中
            # end
    acc_class_wise = [predict_true_class[i] / total_class_label[i] for i in range(args.num_classes)]
    acc = correct / total
    acc_3shot_global["head"] = correct_3shot["head"] / \
        (total_3shot["head"] + 1e-10)
    acc_3shot_global["middle"] = correct_3shot["middle"] / \
        (total_3shot["middle"] + 1e-10)
    acc_3shot_global["tail"] = correct_3shot["tail"] / \
        (total_3shot["tail"] + 1e-10)
    return acc, acc_3shot_global

def globaltest_classmean(net, g_head, test_dataset, args, dataset_class=None, head_switch=True):
    global_test_distribution = dataset_class.global_test_distribution
    three_shot_dict, _ = shot_split(global_test_distribution, threshold_3shot=[75, 95])
    correct_3shot = {"head": 0, "middle": 0, "tail": 0}
    total_3shot = {"head": 0, "middle": 0, "tail": 0}
    acc_3shot_global = {"head": None, "middle": None, "tail": None}
    net.eval()
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=100, shuffle=False)
    # 监视真实情况下所有样本的类别分布
    total_class_label = [0 for i in range(args.num_classes)]
    predict_true_class = [0 for i in range(args.num_classes)]

    # 初始化一个字典来存储每个类别的总和和计数
    class_sums = {}
    class_counts = {}
    # features_list = []
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            features = net(images, latent_output=True)
            # features_list.append(features)
            if head_switch == True:
                outputs = g_head(features)
            else:
                outputs = features
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            #####################
            # 计算class mean
                # 遍历每个样本
            for i in range(images.size(0)):
                # 获取样本的类别和特征值
                label = labels[i].item()
                feature = features[i]

                # 更新类别的总和和计数
                if label not in class_sums:
                    class_sums[label] = feature
                    class_counts[label] = 1
                else:
                    class_sums[label] += feature
                    class_counts[label] += 1
        ##############

            # class-wise acc calc
            for i in range(len(labels)):
                total_class_label[int(labels[i])] += 1      # total
                if predicted[i] == labels[i]:
                    predict_true_class[int(labels[i])] += 1

            # start: cal 3shot metrics
            for label in labels:
                if label in three_shot_dict["head"]:
                    total_3shot["head"] += 1
                elif label in three_shot_dict["middle"]:
                    total_3shot["middle"] += 1
                else:
                    total_3shot["tail"] += 1
            for i in range(len(predicted)):
                if predicted[i] == labels[i] and labels[i] in three_shot_dict["head"]:   # 预测正确且在head中
                    correct_3shot["head"] += 1
                # 预测正确且在middle中
                elif predicted[i] == labels[i] and labels[i] in three_shot_dict["middle"]:
                    correct_3shot["middle"] += 1
                # 预测正确且在tail中
                elif predicted[i] == labels[i] and labels[i] in three_shot_dict["tail"]:
                    correct_3shot["tail"] += 1      # 在tail中
            # end
    
    # 计算每个类别的平均值
    class_means = {label: class_sum / class_counts[label] for label, class_sum in class_sums.items()}
    class_norms = {label: torch.norm(mean, p=2) for label, mean in class_means.items()}
    acc_class_wise = [predict_true_class[i] / total_class_label[i] for i in range(args.num_classes)]
    angle = compute_angle(0, 1, class_means)
    # 假设有100个类别

    # 初始化一个全零的矩阵来保存角度
    angle_matrix = torch.zeros(args.num_classes, args.num_classes)

    # 计算所有类别之间的角度
    for i in range(args.num_classes):
        for j in range(i+1, args.num_classes):  # 注意我们只需要计算上三角部分
            angle = compute_angle(i, j, class_means)
            angle_matrix[i, j] = angle
            angle_matrix[j, i] = angle  # 角度是对称的，所以我们也可以填充下三角部分

    # 现在 angle_matrix[i, j] 是类别 i 和类别 j 之间的角度
    # 创建热图
    plt.figure(figsize=(10, 10))
    plt.imshow(angle_matrix, cmap='hot', interpolation='nearest')

    # 添加颜色条
    plt.colorbar()

    # 设置标题和坐标轴标签
    plt.title('Angle Matrix')
    plt.xlabel('Class')
    plt.ylabel('Class')

    # 保存图像
    plt.savefig('angle_matrix.png')


    class_norms = {label: norm.cpu() for label, norm in class_norms.items()}
    # 提取类别标签和L2范数
    labels = list(class_norms.keys())
    norms = list(class_norms.values())

    # 创建条形图
    plt.figure(figsize=(10, 5))
    plt.bar(labels, norms)

    # 设置标题和坐标轴标签
    plt.title('L2 Norms of Class Means')
    plt.xlabel('Class')
    plt.ylabel('L2 Norm')

    # 保存图像
    plt.savefig('class_norms.png')




    # 计算每个类别的L2范数
    class_norms = torch.norm(g_head.weight, dim=1)

    # 计算所有类别之间的角度
    num_classes = g_head.weight.size(0)
    angle_matrix = torch.zeros(num_classes, num_classes)
    for i in range(num_classes):
        for j in range(i+1, num_classes):  # 注意我们只需要计算上三角部分
            angle = compute_angle(i, j, g_head.weight)
            angle_matrix[i, j] = angle
            angle_matrix[j, i] = angle  # 角度是对称的，所以我们也可以填充下三角部分

    # 创建一个不需要梯度的版本并转移到CPU，然后转换为numpy数组
    class_norms = class_norms.detach().cpu().numpy()
    angle_matrix = angle_matrix.detach().cpu().numpy()

    # 创建条形图
    plt.figure(figsize=(10, 5))
    plt.bar(range(num_classes), class_norms)
    plt.title('L2 Norms of Class Vectors')
    plt.xlabel('Class')
    plt.ylabel('L2 Norm')
    plt.savefig('cls_class_norms.png')

    # 创建热图
    plt.figure(figsize=(10, 10))
    plt.imshow(angle_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Angle Matrix')
    plt.xlabel('Class')
    plt.ylabel('Class')
    plt.savefig('cls_angle_matrix.png')
    plt.show()


    class_norms = {label: torch.norm(mean, p=2) for label, mean in class_means.items()}



    acc = correct / total
    acc_3shot_global["head"] = correct_3shot["head"] / \
        (total_3shot["head"] + 1e-10)
    acc_3shot_global["middle"] = correct_3shot["middle"] / \
        (total_3shot["middle"] + 1e-10)
    acc_3shot_global["tail"] = correct_3shot["tail"] / \
        (total_3shot["tail"] + 1e-10)
    return acc, acc_3shot_global




def globaltest_feat_collapse(net, g_head, test_dataset, args, dataset_class=None, head_switch=True):
    global_test_distribution = dataset_class.global_test_distribution
    three_shot_dict, _ = shot_split(global_test_distribution, threshold_3shot=[75, 95])
    correct_3shot = {"head": 0, "middle": 0, "tail": 0}
    total_3shot = {"head": 0, "middle": 0, "tail": 0}
    acc_3shot_global = {"head": None, "middle": None, "tail": None}
    net.eval()
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=100, shuffle=False)
    # 监视真实情况下所有样本的类别分布
    total_class_label = [0 for i in range(args.num_classes)]
    predict_true_class = [0 for i in range(args.num_classes)]

    # 初始化一个字典来存储每个类别的总和和计数
    class_sums = {}
    class_counts = {}
    # features_list = []
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            features = net(images, latent_output=True)
            # features_list.append(features)
            if head_switch == True:
                outputs = g_head(features)
            else:
                outputs = features
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            #####################
            # 计算class mean
                # 遍历每个样本
            for i in range(images.size(0)):
                # 获取样本的类别和特征值
                label = labels[i].item()
                feature = features[i]

                # 更新类别的总和和计数
                if label not in class_sums:
                    class_sums[label] = feature
                    class_counts[label] = 1
                else:
                    class_sums[label] += feature
                    class_counts[label] += 1
        ##############

            # class-wise acc calc
            for i in range(len(labels)):
                total_class_label[int(labels[i])] += 1      # total
                if predicted[i] == labels[i]:
                    predict_true_class[int(labels[i])] += 1

            # start: cal 3shot metrics
            for label in labels:
                if label in three_shot_dict["head"]:
                    total_3shot["head"] += 1
                elif label in three_shot_dict["middle"]:
                    total_3shot["middle"] += 1
                else:
                    total_3shot["tail"] += 1
            for i in range(len(predicted)):
                if predicted[i] == labels[i] and labels[i] in three_shot_dict["head"]:   # 预测正确且在head中
                    correct_3shot["head"] += 1
                # 预测正确且在middle中
                elif predicted[i] == labels[i] and labels[i] in three_shot_dict["middle"]:
                    correct_3shot["middle"] += 1
                # 预测正确且在tail中
                elif predicted[i] == labels[i] and labels[i] in three_shot_dict["tail"]:
                    correct_3shot["tail"] += 1      # 在tail中
            # end
    acc = correct / total
    acc_3shot_global["head"] = correct_3shot["head"] / \
        (total_3shot["head"] + 1e-10)
    acc_3shot_global["middle"] = correct_3shot["middle"] / \
        (total_3shot["middle"] + 1e-10)
    acc_3shot_global["tail"] = correct_3shot["tail"] / \
        (total_3shot["tail"] + 1e-10)
    print("默认inference性能:")
    print(acc, acc_3shot_global)
    
    # 计算每个类别的平均值
    class_means = {label: class_sum / class_counts[label] for label, class_sum in class_sums.items()}

    # T-SNE


    # 对cls进行裁剪
    # 初始化一个字典来存储被设置为0的位置
    zero_positions = {}
    beta = 0.99
    # 遍历 class_means 字典中的每一项
    for class_label, tensor in class_means.items():
        # 对数据进行排序
        sorted_values, _ = torch.sort(tensor)
        # 找到对应于阈值 beta 的值
        # threshold_index = int(len(sorted_values) * beta)
        threshold_index = int(len(sorted_values) * (1-beta))
        threshold_value = sorted_values[threshold_index]
        # 记录所有小于阈值的元素的位置
        # zero_positions[class_label] = (tensor < threshold_value)
        zero_positions[class_label] = (tensor > threshold_value)

        # 将所有小于阈值的元素设置为 0
        # tensor[tensor < threshold_value] = 0
        tensor[tensor > threshold_value] = 0



    # 1. 复制原始 g_head 的 weight 和 bias
    original_weight = g_head.weight.detach().clone()
    original_bias = g_head.bias.detach().clone()
    for key, value in zero_positions.items():
            original_weight[key, :] *= (~value).float()  # 确保数据类型匹配

    # 3. 创建一个新的 Linear 层，并使用修改后的 weight 和 bias 初始化它
    new_g_head = nn.Linear(in_features=512, out_features=100, bias=True)
    new_g_head.weight.data = original_weight
    new_g_head.bias.data = original_bias
    new_g_head = new_g_head.to(args.device)

    # return acc, acc_3shot_global, new_g_head
        # 裁剪之后的inference
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            features = net(images, latent_output=True)
            # features_list.append(features)
            if head_switch == True:
                outputs = new_g_head(features)
            else:
                outputs = features
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            #####################
            # 计算class mean
                # 遍历每个样本
            for i in range(images.size(0)):
                # 获取样本的类别和特征值
                label = labels[i].item()
                feature = features[i]

                # 更新类别的总和和计数
                if label not in class_sums:
                    class_sums[label] = feature
                    class_counts[label] = 1
                else:
                    class_sums[label] += feature
                    class_counts[label] += 1
        ##############

            # class-wise acc calc
            for i in range(len(labels)):
                total_class_label[int(labels[i])] += 1      # total
                if predicted[i] == labels[i]:
                    predict_true_class[int(labels[i])] += 1

            # start: cal 3shot metrics
            for label in labels:
                if label in three_shot_dict["head"]:
                    total_3shot["head"] += 1
                elif label in three_shot_dict["middle"]:
                    total_3shot["middle"] += 1
                else:
                    total_3shot["tail"] += 1
            for i in range(len(predicted)):
                if predicted[i] == labels[i] and labels[i] in three_shot_dict["head"]:   # 预测正确且在head中
                    correct_3shot["head"] += 1
                # 预测正确且在middle中
                elif predicted[i] == labels[i] and labels[i] in three_shot_dict["middle"]:
                    correct_3shot["middle"] += 1
                # 预测正确且在tail中
                elif predicted[i] == labels[i] and labels[i] in three_shot_dict["tail"]:
                    correct_3shot["tail"] += 1      # 在tail中

    acc = correct / total
    acc_3shot_global["head"] = correct_3shot["head"] / \
        (total_3shot["head"] + 1e-10)
    acc_3shot_global["middle"] = correct_3shot["middle"] / \
        (total_3shot["middle"] + 1e-10)
    acc_3shot_global["tail"] = correct_3shot["tail"] / \
        (total_3shot["tail"] + 1e-10)
    print("裁剪后inference性能:")
    print(acc, acc_3shot_global)
    a = 1

    # # 用class mean作为feature去做global test
    # # 可以看出class mean经过cls能很好被预测，再分析这些class mean的特点吧，如果我裁剪掉class mean中不重要的feat，那sample还能很好地预测嘛
    # count = 0
    # for i in range(100):
    #     value, indice = torch.max(g_head(class_means[i]), 0)
    #     if indice.item() == i:
    #         # print(indice)
    #         count += 1
    # print("class means作为features的预测准确率")
    # print(count/100)

    # # 能看出来大多数的数值都是在0附近
    # plt.hist(class_means[0].cpu().numpy(), bins=50, range=(-2, 2), color='blue', edgecolor='black')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of class_means[0]')
    # plt.yscale('log')  # 对y轴进行对数缩放，如果需要的话
    # # 保存直方图为PNG文件
    # plt.savefig('class_means_0_histogram.png')

    # 初始化一个字典来存储被设置为0的位置
    zero_positions = {}
    beta = 0.4
    # 遍历 class_means 字典中的每一项
    for class_label, tensor in class_means.items():
        # 对数据进行排序
        sorted_values, _ = torch.sort(tensor)
        # 找到对应于阈值 beta 的值
        threshold_index = int(len(sorted_values) * beta)
        threshold_value = sorted_values[threshold_index]
        # 记录所有小于阈值的元素的位置
        zero_positions[class_label] = (tensor < threshold_value)
        # 将所有小于阈值的元素设置为 0
        tensor[tensor < threshold_value] = 0

    # # 裁剪过之后再inference
    # count_after = 0
    # for i in range(100):
    #     value, indice = torch.max(g_head(class_means[i]), 0)
    #     if indice.item() == i:
    #         # print(indice)
    #         count_after += 1
    # print("裁剪后的class means作为features的预测准确率")
    # print(count_after/100)


    # 裁剪之后的inference
    # with torch.no_grad():
    #     correct = 0
    #     total = 0
    #     for images, labels in test_loader:
    #         images = images.to(args.device)
    #         labels = labels.to(args.device)
    #         features = net(images, latent_output=True)

    #         # 根据之前记录的位置信息，将特征置零
    #         for i, label in enumerate(labels):
    #             zero_pos = zero_positions[label.item()]
    #             features[i][zero_pos] = 0


    #         # features_list.append(features)
    #         if head_switch == True:
    #             outputs = g_head(features)
    #         else:
    #             outputs = features
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()

    #         #####################
    #         # 计算class mean
    #             # 遍历每个样本
    #         for i in range(images.size(0)):
    #             # 获取样本的类别和特征值
    #             label = labels[i].item()
    #             feature = features[i]

    #             # 更新类别的总和和计数
    #             if label not in class_sums:
    #                 class_sums[label] = feature
    #                 class_counts[label] = 1
    #             else:
    #                 class_sums[label] += feature
    #                 class_counts[label] += 1
    #     ##############

    #         # class-wise acc calc
    #         for i in range(len(labels)):
    #             total_class_label[int(labels[i])] += 1      # total
    #             if predicted[i] == labels[i]:
    #                 predict_true_class[int(labels[i])] += 1

    #         # start: cal 3shot metrics
    #         for label in labels:
    #             if label in three_shot_dict["head"]:
    #                 total_3shot["head"] += 1
    #             elif label in three_shot_dict["middle"]:
    #                 total_3shot["middle"] += 1
    #             else:
    #                 total_3shot["tail"] += 1
    #         for i in range(len(predicted)):
    #             if predicted[i] == labels[i] and labels[i] in three_shot_dict["head"]:   # 预测正确且在head中
    #                 correct_3shot["head"] += 1
    #             # 预测正确且在middle中
    #             elif predicted[i] == labels[i] and labels[i] in three_shot_dict["middle"]:
    #                 correct_3shot["middle"] += 1
    #             # 预测正确且在tail中
    #             elif predicted[i] == labels[i] and labels[i] in three_shot_dict["tail"]:
    #                 correct_3shot["tail"] += 1      # 在tail中

    # acc = correct / total
    # acc_3shot_global["head"] = correct_3shot["head"] / \
    #     (total_3shot["head"] + 1e-10)
    # acc_3shot_global["middle"] = correct_3shot["middle"] / \
    #     (total_3shot["middle"] + 1e-10)
    # acc_3shot_global["tail"] = correct_3shot["tail"] / \
    #     (total_3shot["tail"] + 1e-10)
    # print("裁剪后inference性能:")
    # print(acc, acc_3shot_global)


    # 初始化用于存储方差和均值的变量
    variance_zero_pos = 0.0
    variance_non_zero_pos = 0.0
    mean_zero_pos = 0.0
    mean_non_zero_pos = 0.0
    count_zero_pos = 0
    count_non_zero_pos = 0
    class_means = {label: class_sum / class_counts[label] for label, class_sum in class_sums.items()}
    # 遍历 test_loader 以获取所有的特征和标签
    for images, labels in test_loader:
        images = images.to(args.device)
        labels = labels.to(args.device)
        features = net(images, latent_output=True)
        
        # 遍历每一个样本和它的标签
        for i, label in enumerate(labels):
            label_item = label.item()
            
            # 获取对应标签的 class_means 和 zero_positions
            class_mean = class_means[label_item]
            zero_pos = zero_positions[label_item]
            
            # 计算方差和均值
            diff = torch.abs(features[i] - class_mean)
            diff_zero_pos = diff[zero_pos]
            diff_non_zero_pos = diff[~zero_pos]
            
            variance_zero_pos += torch.sum(diff_zero_pos ** 2).item()
            variance_non_zero_pos += torch.sum(diff_non_zero_pos ** 2).item()
            
            mean_zero_pos += torch.sum(diff_zero_pos).item()
            mean_non_zero_pos += torch.sum(diff_non_zero_pos).item()
            
            count_zero_pos += len(diff_zero_pos)
            count_non_zero_pos += len(diff_non_zero_pos)

    # 计算平均方差和平均均值
    avg_variance_zero_pos = variance_zero_pos / count_zero_pos if count_zero_pos > 0 else 0.0
    avg_variance_non_zero_pos = variance_non_zero_pos / count_non_zero_pos if count_non_zero_pos > 0 else 0.0

    avg_mean_zero_pos = mean_zero_pos / count_zero_pos if count_zero_pos > 0 else 0.0
    avg_mean_non_zero_pos = mean_non_zero_pos / count_non_zero_pos if count_non_zero_pos > 0 else 0.0

    # 计算相对方差
    relative_variance_zero_pos = avg_variance_zero_pos / (avg_mean_zero_pos ** 2) if avg_mean_zero_pos != 0 else 0.0
    relative_variance_non_zero_pos = avg_variance_non_zero_pos / (avg_mean_non_zero_pos ** 2) if avg_mean_non_zero_pos != 0 else 0.0

    print(f"Relative variance at zero_positions: {relative_variance_zero_pos}")
    print(f"Relative variance at non-zero_positions: {relative_variance_non_zero_pos}")




    import matplotlib.pyplot as plt
    import numpy as np

    # 初始化用于存储第一个类的所有特征的列表
    features_list = []

    # 假设第一个类的标签是 0
    first_class_label = 50

    # 遍历 test_loader 以获取所有的特征和标签
    for images, labels in test_loader:
        images = images.to(args.device)
        labels = labels.to(args.device)
        features = net(images, latent_output=True)
        
        # 找到属于第一个类的特征
        indices = (labels == first_class_label).nonzero(as_tuple=True)[0]
        first_class_features = features[indices]
        
        features_list.append(first_class_features.cpu().detach().numpy())

    # 将所有批次的数据合并成一个数组
    all_features = np.concatenate(features_list, axis=0)

    # 计算均值和方差
    mean_features = np.mean(all_features, axis=0)
    variance_features = np.var(all_features, axis=0)

    # 计算相对方差
    relative_variance = variance_features / (mean_features ** 2)
    relative_variance[np.isnan(relative_variance)] = 0  # 处理除以零的情况

    # 对 mean_features 和 relative_variance 进行排序
    sorted_indices = np.argsort(mean_features)[::-1]
    sorted_mean_features = mean_features[sorted_indices]
    sorted_relative_variance = relative_variance[sorted_indices]


    cls_zero_positions = g_head.weight.cpu().detach().numpy()[first_class_label] == 0
    cls_zero_positions = cls_zero_positions[sorted_indices]

    # 绘制图形
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Sorted Feature Index')
    ax1.set_ylabel('Sorted Mean Features', color='tab:blue')
    ax1.plot(sorted_mean_features, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # 添加权重为0的位置的灰色阴影
    count = 0
    for i in range(len(cls_zero_positions)):
        if cls_zero_positions[i] == True:
            if i < 256:
                count += 1
            ax1.axvspan(i-0.5, i+0.5, facecolor='gray', alpha=0.5)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Sorted Relative Variance', color='tab:red')
    ax2.plot(sorted_relative_variance, color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()
    plt.title('Sorted Mean Features and Relative Variance for Class 0')

    # 保存图像
    fig.savefig('sorted_mean_and_relative_variance.png')


    # 画的是featrues的均值和方差
    # import matplotlib.pyplot as plt
    # import numpy as np

    # # 初始化用于存储第一个类的所有特征和误差的列表
    # features_list = []
    # error_list = []

    # # 假设第一个类的标签是 0
    # first_class_label = 0

    # # 遍历 test_loader 以获取所有的特征和标签
    # for images, labels in test_loader:
    #     images = images.to(args.device)
    #     labels = labels.to(args.device)
    #     features = net(images, latent_output=True)
        
    #     # 找到属于第一个类的特征
    #     indices = (labels == first_class_label).nonzero(as_tuple=True)[0]
    #     first_class_features = features[indices]
        
    #     # 计算与第一个类的 class_means 的误差
    #     class_mean = class_means[first_class_label]
    #     errors = torch.abs(first_class_features - class_mean)
        
    #     features_list.append(first_class_features.cpu().detach().numpy())
    #     error_list.append(errors.cpu().detach().numpy())

    # # 将所有批次的数据合并成一个数组
    # all_features = np.concatenate(features_list, axis=0)
    # all_errors = np.concatenate(error_list, axis=0)

    # # 计算均值和误差的均值
    # mean_features = np.mean(all_features, axis=0)
    # mean_errors = np.mean(all_errors, axis=0)

    # # 对 mean_features 进行排序，并相应地重新排列 mean_errors
    # sorted_indices = np.argsort(mean_features)[::-1]
    # mean_features = mean_features[sorted_indices]
    # mean_errors = mean_errors[sorted_indices]

    # # 绘制图形
    # fig, ax1 = plt.subplots()

    # ax1.set_xlabel('Sorted Feature Index')
    # ax1.set_ylabel('Sorted Mean Features', color='tab:blue')
    # ax1.plot(mean_features, color='tab:blue')
    # ax1.tick_params(axis='y', labelcolor='tab:blue')

    # ax2 = ax1.twinx()
    # ax2.set_ylabel('Sorted Mean Errors', color='tab:red')
    # ax2.plot(mean_errors, color='tab:red')
    # ax2.tick_params(axis='y', labelcolor='tab:red')

    # fig.tight_layout()
    # plt.title('Sorted Mean Features and Errors for Class 0')

    # # 保存图像
    # fig.savefig('sorted_mean_and_errors.png')



    # 画出output的方差均值
    # import matplotlib.pyplot as plt
    # import numpy as np

    # # 初始化用于存储第一个类的所有输出的列表
    # outputs_list = []

    # # 假设第一个类的标签是 0
    # first_class_label = 0

    # # 遍历 test_loader 以获取所有的输出和标签
    # with torch.no_grad():
    #     for images, labels in test_loader:
    #         images = images.to(args.device)
    #         labels = labels.to(args.device)
    #         features = net(images, latent_output=True)
            
    #         if head_switch == True:
    #             outputs = g_head(features)
    #         else:
    #             outputs = features
            
    #         # 找到属于第一个类的输出
    #         indices = (labels == first_class_label).nonzero(as_tuple=True)[0]
    #         first_class_outputs = outputs[indices]
            
    #         outputs_list.append(first_class_outputs.cpu().detach().numpy())

    # # 将所有批次的数据合并成一个数组
    # all_outputs = np.concatenate(outputs_list, axis=0)

    # # 计算均值和方差
    # outputs_mean = np.mean(all_outputs, axis=0)
    # outputs_variance = np.var(all_outputs, axis=0)

    # # 对 outputs_mean 进行排序，并获取排序后的索引
    # sorted_indices = np.argsort(outputs_mean)[::-1]  # 从大到小排序

    # # 使用排序后的索引重新排列 outputs_mean 和 outputs_variance
    # sorted_outputs_mean = outputs_mean[sorted_indices]
    # sorted_outputs_variance = outputs_variance[sorted_indices]

    # # 绘制图形
    # fig, ax = plt.subplots()

    # ax.set_xlabel('Sorted Output Index')
    # ax.set_ylabel('Sorted Mean Outputs')
    # ax.plot(sorted_outputs_mean, label='Sorted Mean Outputs')
    # ax.fill_between(range(len(sorted_outputs_mean)), sorted_outputs_mean - np.sqrt(sorted_outputs_variance), sorted_outputs_mean + np.sqrt(sorted_outputs_variance), color='gray', alpha=0.5, label='Std Deviation')

    # ax.legend()
    # fig.tight_layout()
    # plt.title('Sorted Mean Outputs and Std Deviation for Class 0')

    # # 保存图像
    # fig.savefig('sorted_mean_outputs_and_std_deviation.png')


    return acc, acc_3shot_global



# 过滤掉一定百分比的features
def globaltest_class_mean_filter(net, g_head, test_dataset, class_means, args, dataset_class=None, head_switch=True):
    global_test_distribution = dataset_class.global_test_distribution
    three_shot_dict, _ = shot_split(global_test_distribution, threshold_3shot=[75, 95])
    correct_3shot = {"head": 0, "middle": 0, "tail": 0}
    total_3shot = {"head": 0, "middle": 0, "tail": 0}
    acc_3shot_global = {"head": None, "middle": None, "tail": None}
    net.eval()
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=100, shuffle=False)
    # 监视真实情况下所有样本的类别分布
    total_class_label = [0 for i in range(args.num_classes)]
    predict_true_class = [0 for i in range(args.num_classes)]


    # 初始化一个字典来存储被设置为0的位置
    zero_positions = {}
    beta = 0.001
    # 遍历 class_means 字典中的每一项
    for class_label, tensor in class_means.items():
        # 对数据进行排序
        sorted_values, _ = torch.sort(tensor)
        # 找到对应于阈值 beta 的值
        threshold_index = int(len(sorted_values) * beta)
        threshold_value = sorted_values[threshold_index]
        # 记录所有小于阈值的元素的位置
        zero_positions[class_label] = (tensor < threshold_value)
        # 将所有小于阈值的元素设置为 0
        tensor[tensor < threshold_value] = 0


    # 1. 复制原始 g_head 的 weight 和 bias
    original_weight = g_head.weight.detach().clone()
    original_bias = g_head.bias.detach().clone()
    for key, value in zero_positions.items():
            original_weight[key, :] *= value.float()  # 确保数据类型匹配

    # 3. 创建一个新的 Linear 层，并使用修改后的 weight 和 bias 初始化它
    new_g_head = nn.Linear(in_features=512, out_features=100, bias=True)
    new_g_head.weight.data = original_weight
    new_g_head.bias.data = original_bias
    new_g_head = new_g_head.to(args.device)

    # 裁剪之后的inference
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            features = net(images, latent_output=True)
            # # 根据之前记录的位置信息，将特征置零
            # for i, label in enumerate(labels):
            #     zero_pos = zero_positions[label.item()]
            #     features[i][zero_pos] = 0
            if head_switch == True:
                # outputs = g_head(features)
                outputs = new_g_head(features)
            else:
                outputs = features
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()



            # class-wise acc calc
            for i in range(len(labels)):
                total_class_label[int(labels[i])] += 1      # total
                if predicted[i] == labels[i]:
                    predict_true_class[int(labels[i])] += 1

            # start: cal 3shot metrics
            for label in labels:
                if label in three_shot_dict["head"]:
                    total_3shot["head"] += 1
                elif label in three_shot_dict["middle"]:
                    total_3shot["middle"] += 1
                else:
                    total_3shot["tail"] += 1
            for i in range(len(predicted)):
                if predicted[i] == labels[i] and labels[i] in three_shot_dict["head"]:   # 预测正确且在head中
                    correct_3shot["head"] += 1
                # 预测正确且在middle中
                elif predicted[i] == labels[i] and labels[i] in three_shot_dict["middle"]:
                    correct_3shot["middle"] += 1
                # 预测正确且在tail中
                elif predicted[i] == labels[i] and labels[i] in three_shot_dict["tail"]:
                    correct_3shot["tail"] += 1      # 在tail中

    acc = correct / total
    acc_3shot_global["head"] = correct_3shot["head"] / \
        (total_3shot["head"] + 1e-10)
    acc_3shot_global["middle"] = correct_3shot["middle"] / \
        (total_3shot["middle"] + 1e-10)
    acc_3shot_global["tail"] = correct_3shot["tail"] / \
        (total_3shot["tail"] + 1e-10)

    return acc, acc_3shot_global


def globaltest_etf(net, g_head, test_dataset, args, dataset_class=None, head_switch=True):
    global_test_distribution = dataset_class.global_test_distribution
    three_shot_dict, _ = shot_split(global_test_distribution, threshold_3shot=[75, 95])
    correct_3shot = {"head": 0, "middle": 0, "tail": 0}
    total_3shot = {"head": 0, "middle": 0, "tail": 0}
    acc_3shot_global = {"head": None, "middle": None, "tail": None}
    net.eval()
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=100, shuffle=False)
    # 监视真实情况下所有样本的类别分布
    total_class_label = [0 for i in range(args.num_classes)]
    predict_true_class = [0 for i in range(args.num_classes)]
    cur_M = g_head.ori_M
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            features = net(images, latent_output=True)
            if head_switch == True:
                outputs = g_head(features)
            else:
                outputs = features
            outputs = torch.matmul(outputs, cur_M)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # class-wise acc calc
            for i in range(len(labels)):
                total_class_label[int(labels[i])] += 1      # total
                if predicted[i] == labels[i]:
                    predict_true_class[int(labels[i])] += 1

            # start: cal 3shot metrics
            for label in labels:
                if label in three_shot_dict["head"]:
                    total_3shot["head"] += 1
                elif label in three_shot_dict["middle"]:
                    total_3shot["middle"] += 1
                else:
                    total_3shot["tail"] += 1
            for i in range(len(predicted)):
                if predicted[i] == labels[i] and labels[i] in three_shot_dict["head"]:   # 预测正确且在head中
                    correct_3shot["head"] += 1
                # 预测正确且在middle中
                elif predicted[i] == labels[i] and labels[i] in three_shot_dict["middle"]:
                    correct_3shot["middle"] += 1
                # 预测正确且在tail中
                elif predicted[i] == labels[i] and labels[i] in three_shot_dict["tail"]:
                    correct_3shot["tail"] += 1      # 在tail中
            # end
    acc_class_wise = [predict_true_class[i] / total_class_label[i] for i in range(args.num_classes)]
    acc = correct / total
    acc_3shot_global["head"] = correct_3shot["head"] / \
        (total_3shot["head"] + 1e-10)
    acc_3shot_global["middle"] = correct_3shot["middle"] / \
        (total_3shot["middle"] + 1e-10)
    acc_3shot_global["tail"] = correct_3shot["tail"] / \
        (total_3shot["tail"] + 1e-10)
    return acc, acc_3shot_global


# global dataset is balanced
def globaltest_GBA_Layer(backbone, classifier, test_dataset, args, dataset_class = None):
    global_test_distribution = dataset_class.global_test_distribution
    three_shot_dict, _ = shot_split(global_test_distribution, threshold_3shot=[75, 95])
    correct_3shot = {"head":0, "middle":0, "tail":0}    #######
    total_3shot = {"head":0, "middle":0, "tail":0} 
    acc_3shot_global = {"head":None, "middle":None, "tail":None}
    backbone.eval()
    classifier.eval()
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)
    # 监视真实情况下所有样本的类别分布
    distri_class_real = [0 for i in range(100)]
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            feat = backbone(images)
            outputs = classifier(feat)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # start: cal 3shot metrics
            for label in labels:
                distri_class_real[int(label)] += 1      # 监视真实情况下所有样本的类别分布
                if label in three_shot_dict["head"]:
                    total_3shot["head"] += 1
                elif label in three_shot_dict["middle"]:
                    total_3shot["middle"] += 1
                else:
                    total_3shot["tail"] += 1
            for i in range(len(predicted)):
                if predicted[i] == labels[i] and labels[i] in three_shot_dict["head"]:   # 预测正确且在head中
                    correct_3shot["head"] += 1
                elif predicted[i] == labels[i] and labels[i] in three_shot_dict["middle"]:   # 预测正确且在middle中
                    correct_3shot["middle"] += 1
                elif predicted[i] == labels[i] and labels[i] in three_shot_dict["tail"]:   # 预测正确且在tail中
                    correct_3shot["tail"] += 1      # 在tail中
            # end 

    acc = correct / total
    acc_3shot_global["head"] = correct_3shot["head"] / (total_3shot["head"] + 1e-10)
    acc_3shot_global["middle"] = correct_3shot["middle"] / (total_3shot["middle"] + 1e-10)
    acc_3shot_global["tail"] = correct_3shot["tail"] / (total_3shot["tail"] + 1e-10)
    return acc, acc_3shot_global

def localtest_villina(net, test_dataset, dataset_class, idxs, user_id):
    from sklearn.metrics import f1_score
    import copy
    args = dataset_class.get_args()
    net.eval()
    test_loader = torch.utils.data.DataLoader(DatasetSplit(
        test_dataset, idxs), batch_size=args.local_bs, shuffle=False)

    # get overall distribution
    # class_distribution = [0 for _ in range(10000)]  # 10000 >
    # for images, labels in test_loader:
    #     labels = labels.tolist()
    class_distribution_dict = {}

    class_distribution = dataset_class.local_test_distribution[user_id]

    # class_distribution = class_distribution + 100*[0]
    # zero_classes = np.where(class_distribution == 0)[0]
    # for i in zero_classes:
    #     net.linear.weight.data[i, :] = -1e10

    three_shot_dict, _ = shot_split(
        class_distribution, threshold_3shot=[75, 95])
    # three_shot_dict: {"head":[], "middle":[], "tail":[]}   # containg the class id of head, middle and tail respectively

    ypred = []
    ytrue = []
    acc_3shot_local = {"head": None, "middle": None, "tail": None}

    with torch.no_grad():
        correct = 0
        total = 0
        correct_3shot = {"head": 0, "middle": 0, "tail": 0}
        total_3shot = {"head": 0, "middle": 0, "tail": 0}
        correct_classwise = [0 for i in range(args.num_classes)]
        total_classwise = [0 for i in range(args.num_classes)]
        acc_classwise = [0 for i in range(args.num_classes)]
        for images, labels in test_loader:
            # inference
            images = images.to(args.device)
            labels = labels.to(args.device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)

            # calc total metrics
            total += labels.size(0)     # numble of all samples
            # numble of correct predictions
            correct += (predicted == labels).sum().item()
            predicted = predicted.tolist()
            gts = copy.deepcopy(labels)
            gts = gts.tolist()
            ypred.append(predicted)
            ytrue.append(gts)
            # f1 = f1_score(y_true=labels,y_pred=predicted)
            # print(f1)
            # all_f1.append(f1)

            # start: cal 3shot metrics
            for label in labels:
                total_classwise[label.cpu().tolist()] += 1 
                if label in three_shot_dict["head"]:
                    total_3shot["head"] += 1
                elif label in three_shot_dict["middle"]:
                    total_3shot["middle"] += 1
                else:
                    total_3shot["tail"] += 1
            for i in range(len(predicted)):
                if predicted[i] == labels[i]:
                    correct_classwise[labels[i].cpu().tolist()] += 1 
                if predicted[i] == labels[i] and labels[i] in three_shot_dict["head"]:   # 预测正确且在head中
                    correct_3shot["head"] += 1
                # 预测正确且在middle中
                elif predicted[i] == labels[i] and labels[i] in three_shot_dict["middle"]:
                    correct_3shot["middle"] += 1
                # 预测正确且在tail中
                elif predicted[i] == labels[i] and labels[i] in three_shot_dict["tail"]:
                    correct_3shot["tail"] += 1      # 在tail中
            # end



    ypred = sum(ypred, [])
    ytrue = sum(ytrue, [])
    # print(ypred)
    # print(ytrue)
    f1_macro = f1_score(y_true=ytrue, y_pred=ypred, average='macro')
    f1_weighted = f1_score(y_true=ytrue, y_pred=ypred, average='weighted')
    # print(f1)
    # import pdb;pdb.set_trace()
    acc = correct / total

    # start: calc acc_3shot_local
    # acc_3shot_local["head"] = [0, False],False代表无效，平均的时候分母减1
    # 分布不为0，如果没有head，则返回-1，-1不参与平均计算
    acc_3shot_local["head"] = [0, False] if total_3shot["head"] == 0 else [
        (correct_3shot["head"] / total_3shot["head"]), True]
    acc_3shot_local["middle"] = [0, False] if total_3shot["middle"] == 0 else [
        (correct_3shot["middle"] / total_3shot["middle"]), True]
    acc_3shot_local["tail"] = [0, False] if total_3shot["tail"] == 0 else [
        (correct_3shot["tail"] / total_3shot["tail"]), True]
    # end
    for i in range(len(acc_classwise)):
        acc_classwise[i] = correct_classwise[i] / (total_classwise[i]+1e-10)
    # acc = sum(acc_classwise) / len(acc_classwise)
    # print("F1: "+ str(np.mean(f1)))
    return acc, f1_macro, f1_weighted, acc_3shot_local


def localtest(net, g_head, l_head, test_dataset, dataset_class, idxs, user_id):
    from sklearn.metrics import f1_score
    import copy
    args = dataset_class.get_args()
    net.eval()
    test_loader = torch.utils.data.DataLoader(DatasetSplit(
        test_dataset, idxs), batch_size=args.local_bs, shuffle=False)

    # get overall distribution
    # class_distribution = [0 for _ in range(10000)]  # 10000 >
    # for images, labels in test_loader:
    #     labels = labels.tolist()
    class_distribution_dict = {}

    class_distribution = dataset_class.local_test_distribution[user_id]

    # torch.norm(l_head.weight, dim=1)
    p_mode = 1

    if p_mode == 1:
        # 方案1：
        zero_classes = np.where(class_distribution == 0)[0]
        for i in zero_classes:
            g_head.weight.data[i, :] = -1e10
            l_head.weight.data[i, :] = -1e10
    elif p_mode == 2:
        # 方案2：
        norm = torch.norm(l_head.weight, p=2, dim=1)
        # 将g_head.weight转换为torch.nn.Parameter类型
        g_head.weight = nn.Parameter(g_head.weight * norm.unsqueeze(1))
    elif p_mode == 3:
        # 将class_distribution转换为PyTorch的Tensor
        class_distribution_tensor = torch.from_numpy(class_distribution)
        # 由于g_head.weight的形状是[100, 512]，而class_distribution的形状是[100,]，
        # 所以我们需要将class_distribution扩展为[100, 1]，以便进行元素级别的乘法
        class_distribution_tensor = class_distribution_tensor.view(-1, 1)
        # 将class_distribution_tensor移动到与g_head.weight相同的设备上
        class_distribution_tensor = class_distribution_tensor.to(g_head.weight.device)
        # 进行元素级别的乘法
        g_head.weight = nn.Parameter(g_head.weight * class_distribution_tensor)
    elif p_mode == 4:
        # 将权重和偏置相加，并转换为torch.nn.Parameter
        g_head.weight = nn.Parameter(g_head.weight + l_head.weight)
        g_head.bias = nn.Parameter(g_head.bias + l_head.bias)
    elif p_mode == 5:
        g_head.weight = nn.Parameter(g_head.weight * l_head.weight)
        # g_head.bias = nn.Parameter(g_head.bias * l_head.bias)



    three_shot_dict, _ = shot_split(
        class_distribution, threshold_3shot=[75, 95])
    # three_shot_dict: {"head":[], "middle":[], "tail":[]}   # containg the class id of head, middle and tail respectively
    
    ypred = []
    ytrue = []
    acc_3shot_local = {"head": None, "middle": None, "tail": None}

    with torch.no_grad():
        correct = 0
        total = 0
        correct_3shot = {"head": 0, "middle": 0, "tail": 0}
        total_3shot = {"head": 0, "middle": 0, "tail": 0}
        correct_classwise = [0 for i in range(args.num_classes)]
        total_classwise = [0 for i in range(args.num_classes)]
        acc_classwise = [0 for i in range(args.num_classes)]
        for images, labels in test_loader:
            # inference
            images = images.to(args.device)
            labels = labels.to(args.device)
            features = net(images, latent_output=True)

            if p_mode != 8:
                outputs = g_head(features) 
                # outputs = g_head(features) 
                _, predicted = torch.max(outputs.data, 1)
            else:
                 # use l_head for initial prediction
                l_outputs = l_head(features)
                
                # select top 30% classes
                top_30_percent = int(0.1 * l_outputs.size(1))
                _, top_classes = l_outputs.topk(top_30_percent, dim=1)
                
                # create a mask for selected classes
                mask = torch.zeros_like(l_outputs).scatter_(1, top_classes, 1).bool()
                
                # use g_head for final prediction
                g_outputs = g_head(features)
                
                # apply mask to g_head outputs
                masked_g_outputs = g_outputs.masked_fill(~mask, float('-inf')) 
                
                _, predicted = torch.max(masked_g_outputs.data, 1)

            # calc total metrics
            total += labels.size(0)     # numble of all samples
            # numble of correct predictions
            correct += (predicted == labels).sum().item()
            predicted = predicted.tolist()
            gts = copy.deepcopy(labels)
            gts = gts.tolist()
            ypred.append(predicted)
            ytrue.append(gts)
            # f1 = f1_score(y_true=labels,y_pred=predicted)
            # print(f1)
            # all_f1.append(f1)

            # start: cal 3shot metrics
            for label in labels:
                total_classwise[label.cpu().tolist()] += 1 
                if label in three_shot_dict["head"]:
                    total_3shot["head"] += 1
                elif label in three_shot_dict["middle"]:
                    total_3shot["middle"] += 1
                else:
                    total_3shot["tail"] += 1
            for i in range(len(predicted)):
                if predicted[i] == labels[i]:
                    correct_classwise[labels[i].cpu().tolist()] += 1 
                if predicted[i] == labels[i] and labels[i] in three_shot_dict["head"]:   # 预测正确且在head中
                    correct_3shot["head"] += 1
                # 预测正确且在middle中
                elif predicted[i] == labels[i] and labels[i] in three_shot_dict["middle"]:
                    correct_3shot["middle"] += 1
                # 预测正确且在tail中
                elif predicted[i] == labels[i] and labels[i] in three_shot_dict["tail"]:
                    correct_3shot["tail"] += 1      # 在tail中
            # end



    ypred = sum(ypred, [])
    ytrue = sum(ytrue, [])
    # print(ypred)
    # print(ytrue)
    f1_macro = f1_score(y_true=ytrue, y_pred=ypred, average='macro')
    f1_weighted = f1_score(y_true=ytrue, y_pred=ypred, average='weighted')
    # print(f1)
    # import pdb;pdb.set_trace()
    acc = correct / total

    # start: calc acc_3shot_local
    # acc_3shot_local["head"] = [0, False],False代表无效，平均的时候分母减1
    # 分布不为0，如果没有head，则返回-1，-1不参与平均计算
    acc_3shot_local["head"] = [0, False] if total_3shot["head"] == 0 else [
        (correct_3shot["head"] / total_3shot["head"]), True]
    acc_3shot_local["middle"] = [0, False] if total_3shot["middle"] == 0 else [
        (correct_3shot["middle"] / total_3shot["middle"]), True]
    acc_3shot_local["tail"] = [0, False] if total_3shot["tail"] == 0 else [
        (correct_3shot["tail"] / total_3shot["tail"]), True]
    # end
    for i in range(len(acc_classwise)):
        acc_classwise[i] = correct_classwise[i] / (total_classwise[i]+1e-10)
    # acc = sum(acc_classwise) / len(acc_classwise)
    # print("F1: "+ str(np.mean(f1)))
    return acc, f1_macro, f1_weighted, acc_3shot_local


def localtest_etf(net, g_head, l_head, test_dataset, dataset_class, idxs, user_id):
    from sklearn.metrics import f1_score
    import copy
    args = dataset_class.get_args()
    net.eval()
    test_loader = torch.utils.data.DataLoader(DatasetSplit(
        test_dataset, idxs), batch_size=args.local_bs, shuffle=False)

    # get overall distribution
    # class_distribution = [0 for _ in range(10000)]  # 10000 >
    # for images, labels in test_loader:
    #     labels = labels.tolist()
    class_distribution_dict = {}

    class_distribution = dataset_class.local_test_distribution[user_id]


    p_mode = 1
    cur_M = g_head.ori_M

    if p_mode == 1:
        a = 1
        # 方案1：
        # zero_classes = np.where(class_distribution == 0)[0]
        # for i in zero_classes:
        #     cur_M[i, :] = -1e10
        #     l_head.weight.data[i, :] = -1e10
    elif p_mode == 2:
        # 方案2：
        norm = torch.norm(l_head.weight, p=2, dim=1)
        # 将g_head.weight转换为torch.nn.Parameter类型
        g_head.weight = nn.Parameter(g_head.weight * norm.unsqueeze(1))
    elif p_mode == 3:
        # 将class_distribution转换为PyTorch的Tensor
        class_distribution_tensor = torch.from_numpy(class_distribution)
        # 由于g_head.weight的形状是[100, 512]，而class_distribution的形状是[100,]，
        # 所以我们需要将class_distribution扩展为[100, 1]，以便进行元素级别的乘法
        class_distribution_tensor = class_distribution_tensor.view(-1, 1)
        # 将class_distribution_tensor移动到与g_head.weight相同的设备上
        class_distribution_tensor = class_distribution_tensor.to(g_head.weight.device)
        # 进行元素级别的乘法
        g_head.weight = nn.Parameter(g_head.weight * class_distribution_tensor)
    elif p_mode == 4:
        # 将权重和偏置相加，并转换为torch.nn.Parameter
        g_head.weight = nn.Parameter(g_head.weight + l_head.weight)
        g_head.bias = nn.Parameter(g_head.bias + l_head.bias)
    elif p_mode == 5:
        g_head.weight = nn.Parameter(g_head.weight * l_head.weight)
        # g_head.bias = nn.Parameter(g_head.bias * l_head.bias)



    three_shot_dict, _ = shot_split(
        class_distribution, threshold_3shot=[75, 95])
    # three_shot_dict: {"head":[], "middle":[], "tail":[]}   # containg the class id of head, middle and tail respectively
    
    ypred = []
    ytrue = []
    acc_3shot_local = {"head": None, "middle": None, "tail": None}

    with torch.no_grad():
        correct = 0
        total = 0
        correct_3shot = {"head": 0, "middle": 0, "tail": 0}
        total_3shot = {"head": 0, "middle": 0, "tail": 0}
        correct_classwise = [0 for i in range(args.num_classes)]
        total_classwise = [0 for i in range(args.num_classes)]
        acc_classwise = [0 for i in range(args.num_classes)]
        for images, labels in test_loader:
            # inference
            images = images.to(args.device)
            labels = labels.to(args.device)
            features = net(images, latent_output=True)

            if p_mode != 8:
                features = g_head(features)
                # outputs = torch.matmul(g_head(features), cur_M) + l_head(features)
                outputs = torch.matmul(g_head(features), cur_M)
                # outputs = g_head(features) 
                _, predicted = torch.max(outputs.data, 1)
            elif p_mode == 8:
                 # use l_head for initial prediction
                l_outputs = l_head(features)
                
                # select top 30% classes
                top_30_percent = int(0.1 * l_outputs.size(1))
                _, top_classes = l_outputs.topk(top_30_percent, dim=1)
                
                # create a mask for selected classes
                mask = torch.zeros_like(l_outputs).scatter_(1, top_classes, 1).bool()
                
                # use g_head for final prediction
                g_outputs = g_head(features)
                
                # apply mask to g_head outputs
                masked_g_outputs = g_outputs.masked_fill(~mask, float('-inf')) 
                
                _, predicted = torch.max(masked_g_outputs.data, 1)

            # calc total metrics
            total += labels.size(0)     # numble of all samples
            # numble of correct predictions
            correct += (predicted == labels).sum().item()
            predicted = predicted.tolist()
            gts = copy.deepcopy(labels)
            gts = gts.tolist()
            ypred.append(predicted)
            ytrue.append(gts)
            # f1 = f1_score(y_true=labels,y_pred=predicted)
            # print(f1)
            # all_f1.append(f1)

            # start: cal 3shot metrics
            for label in labels:
                total_classwise[label.cpu().tolist()] += 1 
                if label in three_shot_dict["head"]:
                    total_3shot["head"] += 1
                elif label in three_shot_dict["middle"]:
                    total_3shot["middle"] += 1
                else:
                    total_3shot["tail"] += 1
            for i in range(len(predicted)):
                if predicted[i] == labels[i]:
                    correct_classwise[labels[i].cpu().tolist()] += 1 
                if predicted[i] == labels[i] and labels[i] in three_shot_dict["head"]:   # 预测正确且在head中
                    correct_3shot["head"] += 1
                # 预测正确且在middle中
                elif predicted[i] == labels[i] and labels[i] in three_shot_dict["middle"]:
                    correct_3shot["middle"] += 1
                # 预测正确且在tail中
                elif predicted[i] == labels[i] and labels[i] in three_shot_dict["tail"]:
                    correct_3shot["tail"] += 1      # 在tail中
            # end



    ypred = sum(ypred, [])
    ytrue = sum(ytrue, [])
    # print(ypred)
    # print(ytrue)
    f1_macro = f1_score(y_true=ytrue, y_pred=ypred, average='macro')
    f1_weighted = f1_score(y_true=ytrue, y_pred=ypred, average='weighted')
    # print(f1)
    # import pdb;pdb.set_trace()
    acc = correct / total

    # start: calc acc_3shot_local
    # acc_3shot_local["head"] = [0, False],False代表无效，平均的时候分母减1
    # 分布不为0，如果没有head，则返回-1，-1不参与平均计算
    acc_3shot_local["head"] = [0, False] if total_3shot["head"] == 0 else [
        (correct_3shot["head"] / total_3shot["head"]), True]
    acc_3shot_local["middle"] = [0, False] if total_3shot["middle"] == 0 else [
        (correct_3shot["middle"] / total_3shot["middle"]), True]
    acc_3shot_local["tail"] = [0, False] if total_3shot["tail"] == 0 else [
        (correct_3shot["tail"] / total_3shot["tail"]), True]
    # end
    for i in range(len(acc_classwise)):
        acc_classwise[i] = correct_classwise[i] / (total_classwise[i]+1e-10)
    # acc = sum(acc_classwise) / len(acc_classwise)
    # print("F1: "+ str(np.mean(f1)))
    return acc, f1_macro, f1_weighted, acc_3shot_local


def localtest_vallina(net, test_dataset, dataset_class, idxs, user_id):
    from sklearn.metrics import f1_score
    import copy
    args = dataset_class.get_args()
    net.eval()
    test_loader = torch.utils.data.DataLoader(DatasetSplit(
        test_dataset, idxs), batch_size=args.local_bs, shuffle=False)


    ypred = []
    ytrue = []
    acc_3shot_local = {"head": None, "middle": None, "tail": None}

    with torch.no_grad():
        correct = 0
        total = 0
        correct_3shot = {"head": 0, "middle": 0, "tail": 0}
        total_3shot = {"head": 0, "middle": 0, "tail": 0}
        correct_classwise = [0 for i in range(args.num_classes)]
        total_classwise = [0 for i in range(args.num_classes)]
        acc_classwise = [0 for i in range(args.num_classes)]
        for images, labels in test_loader:
            # inference
            images = images.to(args.device)
            labels = labels.to(args.device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)

            # calc total metrics
            total += labels.size(0)     # numble of all samples
            # numble of correct predictions
            correct += (predicted == labels).sum().item()
            predicted = predicted.tolist()
            gts = copy.deepcopy(labels)
            gts = gts.tolist()
            ypred.append(predicted)
            ytrue.append(gts)
            # f1 = f1_score(y_true=labels,y_pred=predicted)


    ypred = sum(ypred, [])
    ytrue = sum(ytrue, [])
    # print(ypred)
    # print(ytrue)
    f1_macro = f1_score(y_true=ytrue, y_pred=ypred, average='macro')
    f1_weighted = f1_score(y_true=ytrue, y_pred=ypred, average='weighted')
    # print(f1)
    # import pdb;pdb.set_trace()
    acc = correct / total
    return acc, f1_macro, f1_weighted, acc_3shot_local




def calculate_metrics(pred_np, seg_np):
    # pred_np: B,N
    # seg_np: B,N
    b = len(pred_np)
    all_f1 = []
    all_sensitivity = []
    all_specificity = []
    all_ppv = []
    all_npv = []
    for i in range(b):

        f1 = metrics.f1_score(seg_np[i], pred_np[i], average='macro')

        # confusion_matrix = metrics.confusion_matrix(seg_np[i], pred_np[i])  # n_class * n_class(<=17)
        # FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)   # n_class，
        # FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)   # n_class，
        # TP = np.diag(confusion_matrix)                                  # n_class，
        # TN = confusion_matrix.sum() - (FP + FN + TP)                    # n_class，

        # TPR = []
        # PPV = []
        # for j in range(len(TP)):
        #     if (TP[j] + FN[j]) == 0:
        #         TPR.append(1)
        #     else:
        #         TPR.append(TP[j] / (TP[j] + FN[j]))
        # for j in range(len(TP)):
        #     if (TP[j] + FP[j]) == 0:
        #         PPV.append(1)
        #     else:
        #         PPV.append(TP[j] / (TP[j] + FP[j]))
        # # # Sensitivity, hit rate, recall, or true positive rate
        # # TPR = TP / (TP + FN)
        # # Specificity or true negative rate
        # TNR = TN / (TN + FP)
        # # # Precision or positive predictive value
        # # PPV = TP / (TP + FP)
        # # Negative predictive value
        # NPV = TN / (TN + FN)

        all_f1.append(f1)
        # all_ppv.append(np.mean(PPV))
        # all_npv.append(np.mean(NPV))
        # all_sensitivity.append(np.mean(TPR))
        # all_specificity.append(np.mean(TNR))
    # return all_f1, all_ppv, all_npv, all_sensitivity, all_specificity  # B,
    return all_f1




def compute_angle(label1, label2, class_means):
    # 获取类别的平均特征值
    mean1 = class_means[label1]
    mean2 = class_means[label2]

    # 计算夹角
    dot_product = torch.dot(mean1, mean2)
    norm1 = torch.norm(mean1)
    norm2 = torch.norm(mean2)
    cos_theta = dot_product / (norm1 * norm2)
    theta = torch.acos(cos_theta)
    return torch.rad2deg(theta)