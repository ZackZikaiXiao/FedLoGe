import copy
from pydoc import cli
import torch
import numpy as np


def dispatch_fedbn(w_locals, w_glob):
    # for param_tensor in w_glob: # 字典的遍历默认是遍历 key，所以param_tensor实际上是键值
    #     print(param_tensor,'\t',w_glob[param_tensor].size())


    for param_tensor in w_glob: 
        if 'bn' in param_tensor:    # batch normalization layer
            continue
        else:
            for client_id in range(len(w_locals)):
                w_locals[client_id][param_tensor] = w_glob[param_tensor]
            # print(param_tensor,'\t',w_glob[param_tensor].size())


    # w_avg = copy.deepcopy(w_glob[0])
    # for k in w_avg.keys():
    #     #print('k',k)
    #     for i in range(1, len(w)):
    #         #print('i',i)
    #         w_avg[k] += w[i][k]
    #         #print(w[i][k])
    #     #w_avg[k] = torch.div(w_avg[k], len(w))
    #     w_avg[k] = w_avg[k] / len(w)

    # prefix = w_glob.__class__.__name__
    # for name, module in w_glob.named_modules():
    #     try:
    #         items = module._modules.items()
    #         assert(len(items))
    #     except:
    #         print(prefix+'.'+name, module)

    return w_locals

def dispatch_fedper(w_locals, w_glob):
    totol_layer = 0
    for param_tensor in w_glob: 
        totol_layer += 1

    iter_layer = 0
    for param_tensor in w_glob: 
        if iter_layer >= totol_layer - 2:    # classifier for personalization
            break
        else:
            for client_id in range(len(w_locals)):
                w_locals[client_id][param_tensor] = w_glob[param_tensor]
            # print(param_tensor,'\t',w_glob[param_tensor].size())
        iter_layer += 1
    return w_locals