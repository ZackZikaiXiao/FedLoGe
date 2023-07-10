import numpy as np
import torch
import torch.nn.functional as F
import math
import copy
from scipy.spatial.distance import cdist


def add_noise(args, dataset, dict_users):
    np.random.seed(args.seed)

    gamma_s = np.random.binomial(1, args.level_n_system, args.num_users)
    gamma_c_initial = np.random.rand(args.num_users)
    gamma_c_initial = (1 - args.level_n_lowerb) * gamma_c_initial + args.level_n_lowerb
    gamma_c = gamma_s * gamma_c_initial

    y_train = np.array(dataset.targets)
    y_train_noisy = np.array(dataset.targets)

    real_noise_level = np.zeros(args.num_users)
    for i in np.where(gamma_c > 0)[0]:
        sample_idx = np.array(list(dict_users[i]))
        prob = np.random.rand(len(sample_idx))
        noisy_idx = np.where(prob <= gamma_c[i])[0]
        y_train_noisy[sample_idx[noisy_idx]] = np.random.randint(0, 10, len(noisy_idx))
        noise_ratio = np.mean(y_train[sample_idx] != y_train_noisy[sample_idx])
        print("Client %d, noise level: %.4f (%.4f), real noise ratio: %.4f" % (
            i, gamma_c[i], gamma_c[i] * 0.9, noise_ratio))
        real_noise_level[i] = noise_ratio
    return (y_train_noisy, gamma_s, real_noise_level)


# def softmax(x):
#     return np.exp(x) / np.exp(x).sum(axis=1).reshape([-1,1])


def get_output(loader, net, args, latent=False, criterion=None):
    net.eval()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(args.device)
            labels = labels.to(args.device)
            labels = labels.long()
            if latent == False:
                if args.model == "googlenet":
                    outputs = net(images).logits
                else:
                    outputs = net(images)
                # outputs = net(images)
                outputs = F.softmax(outputs, dim=1)
            else:
                outputs = net(images, True)
            loss = criterion(outputs, labels)
            if i == 0:
                output_whole = np.array(outputs.cpu())
                loss_whole = np.array(loss.cpu())
            else:
                output_whole = np.concatenate((output_whole, outputs.cpu()), axis=0)
                loss_whole = np.concatenate((loss_whole, loss.cpu()), axis=0)

    # if latent==False:
    #     output_whole = softmax(output_whole)
    if criterion is not None:
        return output_whole, loss_whole
    else:
        return output_whole


def lid_term(X, batch, k=20):
    eps = 1e-6
    X = np.asarray(X, dtype=np.float32)

    batch = np.asarray(batch, dtype=np.float32)
    f = lambda v: - k / (np.sum(np.log(v / (v[-1]+eps)))+eps)
    distances = cdist(X, batch)

    # get the closest k neighbours
    sort_indices = np.apply_along_axis(np.argsort, axis=1, arr=distances)[:, 1:k + 1]
    m, n = sort_indices.shape
    idx = np.ogrid[:m, :n]
    idx[1] = sort_indices
    # sorted matrix
    distances_ = distances[tuple(idx)]
    lids = np.apply_along_axis(f, axis=1, arr=distances_)
    return lids


# 将一个client的样本类别划分成head, middle, tail三部分
def shot_split(class_dtribution, threshold_3shot=[75, 95]):
    threshold_3shot = threshold_3shot  # percentage

    class_distribution = copy.deepcopy(class_dtribution)
    # num2classid2accumu_map[0]:number, num2classid2accumu_map[1]:class, num2classid2accumu_map[2]:cumulative number(percentage)
    map_num2classid2accumu = [[],[],[]]
    for classid in range(len(class_dtribution)):
        map_num2classid2accumu[0].append(class_distribution[classid])
        map_num2classid2accumu[1].append(classid)
    for i in range(len(map_num2classid2accumu[0])):
        for j in range(0,len(map_num2classid2accumu[0]) - i - 1):
            if map_num2classid2accumu[0][j] < map_num2classid2accumu[0][j+1]:
                map_num2classid2accumu[0][j], map_num2classid2accumu[0][j+1] = map_num2classid2accumu[0][j+1], map_num2classid2accumu[0][j]
                map_num2classid2accumu[1][j], map_num2classid2accumu[1][j+1] = map_num2classid2accumu[1][j+1], map_num2classid2accumu[1][j]
    map_num2classid2accumu[2] = (np.cumsum(np.array(map_num2classid2accumu[0]), axis = 0) / sum(map_num2classid2accumu[0]) * 100).tolist()

    three_shot_dict = {"head":[], "middle":[], "tail":[]}   # containg the class id of head, middle and tail respectively
    

    cut1 = 0
    cut2 = 0
    accu_range_auxi = [0] + map_num2classid2accumu[2]
    accu_range = copy.deepcopy(accu_range_auxi)
    for i in range(1, len(accu_range)):
        accu_range[i] = [accu_range_auxi[i-1], accu_range_auxi[i]]
    del accu_range[0]
    for i in range(len(accu_range)):
        if threshold_3shot[0] > accu_range[i][0] and threshold_3shot[0] <= accu_range[i][1]:
            cut1 = i
        if threshold_3shot[1] > accu_range[i][0] and threshold_3shot[1] <= accu_range[i][1]:
            cut2 = i

    for i in range(len(map_num2classid2accumu[1])):
        if i <= cut1:
            three_shot_dict["head"].append(map_num2classid2accumu[1][i])
        elif i > cut1 and i <= cut2:
            three_shot_dict["middle"].append(map_num2classid2accumu[1][i])
        else:
            three_shot_dict["tail"].append(map_num2classid2accumu[1][i])

    ### uncomment to print

    # for i in range(len(map_num2classid2accumu)):
    #     print(map_num2classid2accumu[i])
    # print(three_shot_dict)

    return three_shot_dict, map_num2classid2accumu
