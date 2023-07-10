import  torch
import  torch.nn as nn
import  torch.nn.functional as F
import  torch.optim as optim
from    torchvision import datasets, transforms
import  copy
from    tensorboardX import SummaryWriter
import  random
import  math


class GradientAnalysor():
    def __init__(self, class_num=10):   # 默认10个类别
        self.pos_grad_list = [[] for _ in range(class_num)] # 保存所有的正样本gradient
        self.neg_grad_list = [[] for _ in range(class_num)] # 保存所有负样本的gradient
        self.pos_accum = [0 for _ in range(class_num)]  # 正样本梯度累加
        self.neg_accum = [0 for _ in range(class_num)]  # 负样本梯度累加
        self.pos_neg_ratio = [None for _ in range(class_num)] # ratio做除法
        self.label_counter = [0 for _ in range(class_num)]      # label_counter[i]:类别为i的和样本数目

# gradient_batch[i][j]代表batch中第i个样本的classifier i的输出的梯度，梯度维度为d
# label_batch[i][j]代表batch中第i个样本的类别(0-9)
# 类型为list
# 每个<样本>更新一次
    def update(self, gradient_batch, label_batch): 
        try:
            assert(isinstance(gradient_batch, list))
            assert(isinstance(label_batch, list))
        except:
            return 
        batch_size = len(gradient_batch)
        class_num = len(gradient_batch[0])
        for sample_id in range(batch_size):
            gradient = self.__abs(gradient_batch[sample_id])  # 取绝对值
            # gradient = gradient_batch[sample_id]
            label = label_batch[sample_id]
            self.label_counter[label] += 1 
            for classifier_id in range(len(gradient)):  # 对于每个classifier输出的导数
                # start update 
                if classifier_id == label:  # 正样本
                    self.pos_grad_list[classifier_id].append(gradient[classifier_id])
                    self.pos_accum[classifier_id] += gradient[classifier_id]
                else:   # 负样本
                    self.neg_grad_list[classifier_id].append(gradient[classifier_id])
                    self.neg_accum[classifier_id] += gradient[classifier_id]
                self.pos_neg_ratio[classifier_id] = self.pos_accum[classifier_id] / self.neg_accum[classifier_id] if self.neg_accum[classifier_id] != 0 else None
                # end

    def print_for_debug(self):
        # print("pos grad list:", self.pos_grad_list)  # 保存所有的正样本gradient
        # print("neg grad list:", self.neg_grad_list) # 负样本的gradient
        print("pos accum", self.pos_accum)  # 正样本梯度累加
        print("neg accum", self.neg_accum)  # 负样本梯度累加
        print("pos neg ratio", self.pos_neg_ratio) # ratio做除法
        print("label_counter", self.label_counter)

    # 求范数，默认求2范数
    def __norm(self, list, norm=2):
        sum = 0
        for item in list:
            sum += math.pow(item, norm) 
        return math.pow(sum, 1/norm)

    def __abs(self, list):
        for i in range(len(list)):
            if list[i] >= 0:
                continue
            else:
                list[i] = -list[i]
        return list


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
        self.gradient = grad.cpu().numpy().tolist() # [200, 10] -> [10, 200]
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
        # self.m_count += 1
        # print("model hook", self.m_count)
        # print(grad_input)
        # self.get_gradient = list(grad_input[0])
        # print(self.get_gradient)
        # print('Shape of grad_input', grad_input[0].shape)
        # print('grad_output', grad_output)
        # print(grad_input[0].shape)



# demo: classification on mnist

# torch.manual_seed(0)
batch_size=200
learning_rate=0.01
epochs=100

class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 10),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       # transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True)



if __name__ == "__main__":

    device = torch.device('cuda:0')
    criteon = nn.CrossEntropyLoss().to(device)

    net = MLP().to(device)
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)



    L2_similarityList = []
    cosine_similarityList = []

    # visualization
    # writer = SummaryWriter('runs/explore')


    hookObj = Hook()
    gradAnalysor = GradientAnalysor()

    count = 0
    for epoch in range(epochs):
        # train
        for batch_idx, (data, target) in enumerate(train_loader):
            
            data = data.view(-1, 28*28)
            data, target = data.to(device), target.cuda()
            logits = net(data)
            loss = criteon(logits, target)
            optimizer.zero_grad()    

            # 获取classifier的梯度信息    
            # for name, parms in net.named_parameters():	
            #     a = 1
            # start: analyse grad
            hook_handle = logits.register_hook(hookObj.hook_func_tensor)
            # hook_handle = criteon.register_backward_hook(hookObj.hook_func_operator)       # input就是loss对classifier输出的倒数
            loss.backward() # 此时进入hook
            # print("round count", count)
            if hookObj.has_gradient():
                # print("has, gradient", count)
                gradAnalysor.update(hookObj.get_gradient(), target.cpu().numpy().tolist())  # 输入一个batch的梯度和label
            # end
            optimizer.step()
            # print(batch_idx)
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
            gradAnalysor.print_for_debug()
            count += 1
            hook_handle.remove()
        # handle.remove()
        # print(cal_weight_norm(net.state_dict()))


        # # test
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data = data.view(-1, 28 * 28)
            data, target = data.to(device), target.cuda()
            logits = net(data)
            test_loss += criteon(logits, target).item()

            pred = logits.argmax(dim=1)
            correct += pred.eq(target).float().sum().item()



        test_loss /= len(test_loader.dataset)
        print('\nTest set 1: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        # writer.add_scalar('Net Accuracy', 100. * correct / len(test_loader.dataset), global_step=epoch)


