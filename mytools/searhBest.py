# @Time: 2022/7/26
# @Author: zikai
# @File: search the best result in model test output file

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 全部文件搜索
# # Input
# basefolder = '/home/zikaixiao/zikai/aaFL/fl_framework_lss/temp' # path
# keyword = "weight"


# # basefolder = os.path.join(basefolder_pre,'fl_framework_lss', 'temp')
# filelist = os.listdir(basefolder)
# print(filelist)
# filelist.sort()

# for file in filelist:   # for each file
#     context = []
#     fpath = os.path.join(basefolder,file)
#     f = open(fpath,'r')
#     for line in f.readlines():
#         if keyword in line:     # only the line containing keyword will be added to context
#             context.append(line)
#         else:
#             continue
#     evaluationList = []
#     for i in range(len(context)):
#         evaluationList.append(float(context[i].split(" ")[-2]))
#     print("File \"{0}\" containing keyword \"{1}\" reaches to \"{2}\"".format(file, keyword, max(evaluationList)))


# 单个文件搜索
# keyword = "global 3shot acc"
# # fpath = "/home/zikaixiao/zikai/aaFL/fl_framework_lss/前向后向都没有weight_if100.log"
# # fpath = "/home/zikaixiao/zikai/aaFL/fl_framework_lss/weightmask[0,6]_if100.log"
# # fpath = "/home/zikaixiao/zikai/aaFL/fl_framework_lss/只grad计算有weight_真实梯度_累计加weight_if100.log"
# fpath = "/home/zikai/zikai/aaFL/fl_framework_lss/vallina_balanced_test_if100.log"
# context = []
# # fpath = os.path.join(basefolder,file)
# f = open(fpath, 'r')
# for line in f.readlines():
#     if keyword in line:     # only the line containing keyword will be added to context
#         context.append(line)
#     else:
#         continue
# evaluationList = []
# heads = []
# middles = []
# tails = []
# for i in range(len(context)):
#     heads.append(float(context[i].split(" ")[-8][:-1]))
#     middles.append(float(context[i].split(" ")[-6][:-1]))
#     tails.append(float(context[i].split(" ")[-2][:-1]))
#     evaluationList.append((heads[-1] + middles[-1] + tails[-1]) / 3)
# # search the top acc line
# loc = 0
# max_item = 0
# for i in range(len(evaluationList)):
#     if evaluationList[i] > max_item:
#         max_item = evaluationList[i]
#         loc = i

# print("File \"{0}\" containing keyword \"{1}\" reaches to \"{2}\"".format(
#     fpath.split('/')[-1], keyword, max(evaluationList)))
# print(context[loc])

# 根据总的acc找
# keyword = "local average test a"
keyword = "global test acc"
# fpath = "/home/zikaixiao/zikai/aaFL/fl_framework_lss/前向后向都没有weight_if100.log"
# fpath = "/home/zikaixiao/zikai/aaFL/fl_framework_lss/weightmask[0,6]_if100.log"
fpath = "/home/zikaixiao/zikai/aapfl/pfed_lastest/e_ours.log"
# fpath = "/home/zikaixiao/zikai/aaFL/fl_framework_lss/eql参数8.log"
context = []
# fpath = os.path.join(basefolder,file)
f = open(fpath,'r')
for line in f.readlines():
    if keyword in line:     # only the line containing keyword will be added to context
        context.append(line)
    else:
        continue
evaluationList = []
heads = []
middles = []
tails = []
for i in range(len(context)):
    evaluationList.append(float(context[i].split(" ")[-2]))
# search the top acc line
loc = 0
max_item = 0
for i in range(len(evaluationList)):
    if evaluationList[i] > max_item:
        max_item = evaluationList[i] 
        loc = i
    
print("File \"{0}\" containing keyword \"{1}\" reaches to \"{2}\"".format(fpath.split('/')[-1], keyword, max(evaluationList)))
print(context[loc])





# 根据总的acc找
keyword = "local average test a"
# keyword = "global test acc"
# fpath = "/home/zikaixiao/zikai/aaFL/fl_framework_lss/前向后向都没有weight_if100.log"
# fpath = "/home/zikaixiao/zikai/aaFL/fl_framework_lss/weightmask[0,6]_if100.log"
# fpath = "/home/zikaixiao/zikai/aapfl/fl_gba_cifar100/fedbn_formal.log"
# fpath = "/home/zikaixiao/zikai/aaFL/fl_framework_lss/eql参数8.log"
context = []
# fpath = os.path.join(basefolder,file)
f = open(fpath,'r')
for line in f.readlines():
    if keyword in line:     # only the line containing keyword will be added to context
        context.append(line)
    else:
        continue
evaluationList = []
heads = []
middles = []
tails = []
for i in range(len(context)):
    evaluationList.append(float(context[i].split(" ")[-2]))
# search the top acc line
loc = 0
max_item = 0
for i in range(len(evaluationList)):
    if evaluationList[i] > max_item:
        max_item = evaluationList[i] 
        loc = i
    
print("File \"{0}\" containing keyword \"{1}\" reaches to \"{2}\"".format(fpath.split('/')[-1], keyword, max(evaluationList)))
print(context[loc])