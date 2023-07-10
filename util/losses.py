import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from util import *
from functools import partial
from torch.nn.modules.loss import _Loss


# def focal_loss(input_values, gamma):
#     """Computes the focal loss"""
#     p = torch.exp(-input_values)
#     loss = (1 - p) ** gamma * input_values
#     return loss.mean()


# class FocalLoss(nn.Module):
#     def __init__(self, weight=None, gamma=0.):
#         super(FocalLoss, self).__init__()
#         assert gamma >= 0
#         self.gamma = gamma
#         self.weight = weight

#     def forward(self, input, target):
#         return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)


class LDAMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(
            self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)


class PIDLOSS(nn.Module):
    def __init__(self,
                 use_sigmoid=True,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 num_classes=100,  # 1203 for lvis v1.0, 1230 for lvis v0.5
                 gamma=12,
                 mu=0.8,
                 alpha=4.0,
                 pidmask=["head"],
                 vis_grad=False,
                 test_with_obj=True,
                 device='cpu',
                 class_activation = False):
        super().__init__()
        self.use_sigmoid = True
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.num_classes = num_classes
        self.group = True
        self.hook = Hook()  # 用来改梯度的
        self.controllers = [PID() for _ in range(self.num_classes)]  # 一个类别一个控制器
        self.pidmask = pidmask
        self.class_activation = class_activation
        self.class_acti_mask = None

        # cfg for eqlv2
        self.vis_grad = vis_grad
        self.gamma = gamma
        self.mu = mu
        self.alpha = alpha

        # initial variables
        self.register_buffer('pos_grad', torch.zeros(self.num_classes))
        self.register_buffer('neg_grad', torch.zeros(self.num_classes))
        # At the beginning of training, we set a high value (eg. 100)
        # for the initial gradient ratio so that the weight for pos gradients and neg gradients are 1.
        self.register_buffer('pos_neg', torch.ones(self.num_classes) * 100)
        self.register_buffer('pn_diff', torch.zeros(self.num_classes))
        self.pos_grad = self.pos_grad.to(device)
        self.neg_grad = self.neg_grad.to(device)
        self.pos_neg = self.pos_neg.to(device)
        self.pn_diff = self.pn_diff.to(device)

        self.ce_layer = nn.CrossEntropyLoss()
        self.test_with_obj = test_with_obj

        def _func(x):
            return (10 / 9) / ((1 / 9) + torch.exp(-0.5 * x))
        self.map_func = partial(_func)

    def clear(self):
        self.pos_grad = self.pos_grad - self.pos_grad
        self.neg_grad = self.neg_grad - self.neg_grad
        self.pos_neg = self.pos_neg - self.pos_neg
        self.pn_diff = self.pn_diff - self.pn_diff

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        self.n_i, self.n_c = cls_score.size()

        self.gt_classes = label
        self.pred_class_logits = cls_score
        # hook_handle = cls_score.register_hook(self.hook_func_tensor)

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_i, self.n_c)
            target[torch.arange(self.n_i), gt_classes] = 1
            return target
        # target.shape = [20, 10]
        self.target = expand_label(cls_score, label)     # 生成一个target矩阵

        # PID 
        # weight是啥啊:每个bce有一个loss，一共有[20, 10]个loss
        self.pos_w, self.neg_w = self.get_weight(self.target)
        self.weight = self.pos_w * self.target + self.neg_w * (1 - self.target)

        # class activation，
        if self.class_activation:       # 开启了类激活
            if self.class_acti_mask == None:    # 初始化
                self.class_acti_mask = cls_score.new_ones(self.n_i, self.n_c)
                for i in range(self.n_c):       # 没有mask的class，设置为0 
                    if "head" not in self.pidmask and i in self.head_class:    
                        self.class_acti_mask[torch.arange(self.n_i), i] = 0
                    if "middle" not in self.pidmask and i in self.middle_class:
                        self.class_acti_mask[torch.arange(self.n_i), i] = 0
                    if "tail" not in self.pidmask and i in self.tail_class: 
                        self.class_acti_mask[torch.arange(self.n_i), i] = 0
            else:       # 每次看samples时
                for i in range(label.shape[0]):       
                    one_class = label[i]
                    if "head" not in self.pidmask and one_class in self.head_class:    
                        # print("重要信息：有了类别 ->  ", str(one_class))
                        self.class_acti_mask[torch.arange(self.n_i), one_class] = 1
                        self.controllers[one_class].open()
                    if "middle" not in self.pidmask and one_class in self.middle_class:
                        self.class_acti_mask[torch.arange(self.n_i), one_class] = 1
                        self.controllers[one_class].open()
                        # print("重要信息：有了类别 ->  ", str(one_class))
                    if "tail" not in self.pidmask and one_class in self.tail_class: 
                        self.class_acti_mask[torch.arange(self.n_i), one_class] = 1
                        self.controllers[one_class].open()
                        # print("重要信息：有了类别 ->  ", str(one_class))
            self.weight *= self.class_acti_mask[0:self.n_i, :]
        # 只管tail类别
        # classmask = cls_score.new_zeros(self.n_i, self.n_c)
        # classmask[torch.arange(self.n_i), 6:] = 1   # 就[6-10]是1
        # self.weight *= classmask
        # self.weight[torch.arange(self.n_i), :6] = 1

        cls_loss = F.binary_cross_entropy_with_logits(
            cls_score, self.target, reduction='none')
        # cls_loss = torch.sum(cls_loss * self.weight) / self.n_i
        cls_loss = torch.sum(cls_loss) / self.n_i
        hook_handle = cls_score.register_hook(self.hook_func_tensor)
        # self.collect_grad(cls_score.detach(), self.target.detach(), self.weight.detach())
        # self.print_for_debug()
        # hook_handle.remove()
        return self.loss_weight * cls_loss

    def hook_func_tensor(self, grad):
        # 更改梯度
        grad *= self.weight
        batch_size = grad.shape[0]
        class_nums = grad.shape[1]
        # # 收集梯度: collect_grad可用，这里不再使用
        target_temp = self.target.detach()
        grad_temp = grad.detach()
        grad_temp = torch.abs(grad_temp)

        # 更新accu grad
        # grad_temp *= self.weight
        pos_grad = torch.sum(grad_temp * target_temp, dim=0)
        neg_grad = torch.sum(grad_temp * (1 - target_temp), dim=0)
        self.pos_grad += pos_grad
        self.neg_grad += neg_grad
        self.pos_neg = self.pos_grad / (self.neg_grad + 1e-20)
        # self.pn_diff = torch.abs(self.pos_grad - self.neg_grad)
        self.pn_diff = self.pos_grad - self.neg_grad

        # for sample_id in range(batch_size):     # 对于每个样本
        #     for classifier_id in range(class_nums):  # 对于每个分类器
        #         if classifier_id == self.gt_classes[sample_id]: # 正样本
        #             grad[sample_id][classifier_id] *= self.pos_w[classifier_id]               # 加权
        #         else:
        #             grad[sample_id][classifier_id] *= self.neg_w[classifier_id]               # 加权
        # # print("真实的值:")
        # print(grad)
        # grad = self.grad

        # 调pid参数用的
        # global glo_itr
        # global writer
        # glo_itr += 1
        # # print(glo_itr)
        # for i in range(10):
        #     writer.add_scalar("diff " + str(i),
        #                       self.pn_diff[i], global_step=glo_itr)

    def set_mask(self, pidmask):
        self.pidmask = pidmask

    def get_3shotclass(self, head_class, middle_class, tail_class):
        self.head_class = head_class
        self.middle_class = middle_class
        self.tail_class = tail_class

    def apply_3shot_mask(self):
        # apply 3shot mask
        # 3shot mask操作，head就不加权重了，不然性能血崩
        if "head" in self.pidmask:
            for i in self.head_class:
                # self.weight[torch.arange(self.n_i), self.head_class[i]] = 1
                self.controllers[i].reset()
                self.controllers[i].close()
        else:
             for i in self.head_class:
                self.controllers[i].reset()
                self.controllers[i].open()

        if "middle" in self.pidmask:
            for i in self.middle_class:
                # self.weight[torch.arange(self.n_i), self.middle_class[i]] = 1
                self.controllers[i].reset()
                self.controllers[i].close()
        else:
             for i in self.middle_class:
                self.controllers[i].reset()
                self.controllers[i].open()

        if "tail" in self.pidmask:
            for i in self.tail_class:
                # self.weight[torch.arange(self.n_i), self.tail_class[i]] = 1
                self.controllers[i].reset()
                self.controllers[i].close()
        else:
            for i in self.tail_class:
                self.controllers[i].reset()
                self.controllers[i].open()

    def apply_class_activation(self):
        if self.class_activation:   # 开启了类激活
            # 关闭没有mask的class pid，在forward阶段按照label打开
            if "head" not in self.pidmask:
                for i in self.head_class:
                    # self.weight[torch.arange(self.n_i), self.head_class[i]] = 1
                    self.controllers[i].reset()
                    self.controllers[i].close()

            if "middle" not in self.pidmask:
                for i in self.middle_class:
                    # self.weight[torch.arange(self.n_i), self.middle_class[i]] = 1
                    self.controllers[i].reset()
                    self.controllers[i].close()

            if "tail" not in self.pidmask:
                for i in self.tail_class:
                    # self.weight[torch.arange(self.n_i), self.tail_class[i]] = 1
                    self.controllers[i].reset()
                    self.controllers[i].close()
   

    def get_weight(self, target):
        # # 每个类别都有一个pos weight和neg weight

        # pos_w = [0 for i in range(self.num_classes)]
        # neg_w = [0 for i in range(self.num_classes)]

        pos_w = target.new_zeros(self.num_classes)      #
        neg_w = target.new_zeros(self.num_classes)
        debug = 11
        for i in range(self.num_classes):       # 对于十个类别中的第i个
            pid_out = self.controllers[i].PID_calc(self.pn_diff[i], 0)
            # 10是一个参数，大了对pos neg weight影响更极端
            if 0 - self.pn_diff[i] > 0:     # neg太多
                pos_w[i] = self.map_func(pid_out)   # 让pos多点
                neg_w[i] = self.map_func(-pid_out)  # 让neg少点
            else:                           # pos太多
                pos_w[i] = self.map_func(pid_out)  # 让pos少点
                neg_w[i] = self.map_func(-pid_out)   # 让neg多点

        debug = 12
        # neg_w = self.map_func(self.pos_neg)
        # pos_w = 1 + self.alpha * (1 - neg_w)
        # neg_w = neg_w.view(1, -1).expand(self.n_i, self.n_c)
        # pos_w = pos_w.view(1, -1).expand(self.n_i, self.n_c)
        return pos_w, neg_w

    def print_for_debug(self):
        # print("pos", self.pos_grad)
        # print("neg", self.neg_grad)
        # print("ratio", self.pos_neg)
        print("diff", self.pn_diff)
        # print("pos_w", self.pos_w)
        # print("neg_w", self.neg_w)
        pass

    def collect_grad(self, cls_score, target, weight=None):
        prob = torch.sigmoid(cls_score)
        grad = target * (prob - 1) + (1 - target) * prob
        grad = torch.abs(grad)
        batch_size = grad.shape[0]
        grad = grad / batch_size
        # do not collect grad for objectiveness branch [:-1] why?
        # pos_grad = torch.sum(grad * target * weight, dim=0)[:-1]
        # neg_grad = torch.sum(grad * (1 - target) * weight, dim=0)[:-1]
        pos_grad = torch.sum(grad * target * weight, dim=0)
        neg_grad = torch.sum(grad * (1 - target) * weight, dim=0)

        # dist.all_reduce(pos_

        self.pos_grad += pos_grad
        self.neg_grad += neg_grad
        self.pos_neg = self.pos_grad / (self.neg_grad + 1e-20)

        # print("计算的值：")
        # print("grad", grad)
        # print("pos_grad", self.pos_grad)
        # print("neg_grad", self.neg_grad)
        # print("pos_neg", self.pos_neg)

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

    def standardization(self, data):
        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)
        return (data - mu) / sigma


class Hook():
    def __init__(self):
        self.m_count = 0    # for debug
        # hook函数中临时变量的保存
        self.input_grad_list = []
        self.output_grad_list = []
        self.gradient = None
        self.gradient_list = []

    def has_gradient(self):
        return self.gradient != None

    def get_gradient(self):
        return self.gradient

    def hook_func_tensor(self, grad):
        grad = copy.deepcopy(grad)
        self.gradient = grad.cpu().numpy().tolist()  # [200, 10] -> [10, 200]
        # print(type(self.gradient))
        # print("tensor hook", self.m_count)
        # print(grad)
        # print(grad.shape)
        self.m_count += 1

    def hook_func_model(self, module, grad_input, grad_output):
        pass
        # print("model hook", )
        # print(module)
        # print('grad_input', grad_input)
        # print('grad_output', grad_output)

    def hook_func_operator(self, module, grad_input, grad_output):
        pass


class PID():
    def __init__(self):
        self.mode = "PID_DELTA"  # PID_POSITION
        self.Kp = 10
        self.Ki = 0.01
        self.Kd = 0.1

        self.max_out = 100  # PID最大输出
        self.max_iout = 100  # PID最大积分输出

        self.set = 0	  # PID目标值
        self.current_value = 0	  # PID当前值

        self.out = 0		# 三项叠加输出
        self.Pout = 0		# 比例项输出r
        self.Iout = 0		# 积分项输出
        self.Dout = 0		# 微分项输出
        # 微分项最近三个值 0最新 1上一次 2上上次
        self.Dbuf = [0, 0, 0]
        # 误差项最近三个值 0最新 1上一次 2上上次
        self.error = [0, 0, 0]
        self.m_open = False # 初始化的时候不激活
        # self.count = 0

    def reset(self):
        self.current_value = 0	  # PID当前值
        self.out = 0		# 三项叠加输出
        self.Pout = 0		# 比例项输出r
        self.Iout = 0		# 积分项输出
        self.Dout = 0		# 微分项输出
        # 微分项最近三个值 0最新 1上一次 2上上次
        self.Dbuf = [0, 0, 0]
        # 误差项最近三个值 0最新 1上一次 2上上次
        self.error = [0, 0, 0]
        self.m_open = False # 初始化的时候不激活


    def open(self):
        self.m_open = True
        
    
    def close(self):
        self.m_open = False
        

    def is_open(self):
        return self.m_open

    def PID_calc(self, current_value, set_value):
        # self.count += 1
        # print(self.count)
        if self.m_open == False:
            return torch.Tensor([0.])    # 不开启pid，返回偏差为0
        # # 判断传入的PID指针不为空
        # 存放过去两次计算的误差值
        self.error[2] = self.error[1]
        self.error[1] = self.error[0]
        # 设定目标值和当前值到结构体成员
        self.set_value = set_value
        self.current_value = current_value
        # 计算最新的误差值
        self.error[0] = set_value - current_value
        # 判断PID设置的模式
        if self.mode == "PID_POSITION":
            # 位置式PID
            # 比例项计算输出
            self.Pout = self.Kp * self.error[0]
            # 积分项计算输出
            self.Iout += self.Ki * self.error[0]
            # 存放过去两次计算的微分误差值
            self.Dbuf[2] = self.Dbuf[1]
            self.Dbuf[1] = self.Dbuf[0]
            # 当前误差的微分用本次误差减去上一次误差来计算
            self.Dbuf[0] = (self.error[0] - self.error[1])
            # 微分项输出
            self.Dout = self.Kd * self.Dbuf[0]
            # 对积分项进行限幅
            self.LimitMax(self.Iout, self.max_iout)
            # 叠加三个输出到总输出
            self.out = self.Pout + self.Iout + self.Dout
            # 对总输出进行限幅
            self.LimitMax(self.out, self.max_out)

        elif self.mode == "PID_DELTA":

            # 增量式PID
            # 以本次误差与上次误差的差值作为比例项的输入带入计算
            self.Pout = self.Kp * (self.error[0] - self.error[1])
            # 以本次误差作为积分项带入计算
            self.Iout = self.Ki * self.error[0]
            # 迭代微分项的数组
            self.Dbuf[2] = self.Dbuf[1]
            self.Dbuf[1] = self.Dbuf[0]
            # 以本次误差与上次误差的差值减去上次误差与上上次误差的差值作为微分项的输入带入计算
            self.Dbuf[0] = self.error[0] - 2.0 * self.error[1] + self.error[2]
            self.Dout = self.Kd * self.Dbuf[0]
            # 叠加三个项的输出作为总输出
            self.out += self.Pout + self.Iout + self.Dout
            # 对总输出做一个先限幅
            self.LimitMax(self.out, self.max_out)

        return self.out

    def LimitMax(self, input, max):
        if input > max:
            input = max
        elif input < -max:
            input = -max


# 支持多分类和二分类
class FocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """
 
    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average
 
        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')
 
        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')
 
    def forward(self, input, target):
        logit = F.softmax(input, dim=1)
 
        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)
 
        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)
 
        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)
 
        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()
 
        gamma = self.gamma
 
        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt
 
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class BalancedSoftmax(_Loss):
    """
    Balanced Softmax Loss
    """
    def __init__(self, freq):
        super(BalancedSoftmax, self).__init__()
        freq = torch.tensor(freq)
        self.sample_per_class = freq

    def forward(self, input, label, reduction='mean'):
        return balanced_softmax_loss(label, input, self.sample_per_class, reduction)


def balanced_softmax_loss(labels, logits, sample_per_class, reduction):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss


def create_loss(freq_path):
    print('Loading Balanced Softmax Loss.')
    return BalancedSoftmax(freq_path)