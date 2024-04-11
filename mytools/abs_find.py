import re

from matplotlib import pyplot as plt
keyword1 = "The value of constant scalar:"

keyword2 = "The value of constant scalar, without abs:"
fpath = "/home/jianhuiwei/rsch/jianhui/FedLoGe/Analysis_experiments_for_abs.log"
fpath2 = "/home/jianhuiwei/rsch/jianhui/FedLoGe/scalars_per_cls_initialized_0.1_without_abs.log"
context1 = []
context2 = []
context3 = []
# fpath = os.path.join(basefolder,file)
f = open(fpath,'r')
for line in f.readlines():
    if keyword1 in line:     # only the line containing keyword will be added to context
        context1.append(line)
    elif keyword2 in line:
        context2.append(line)
    else:
        continue
f = open(fpath2,'r')
for line in f.readlines():
    if keyword1 in line:     # only the line containing keyword will be added to context
        context3.append(line)
result_list1 = []
result_list2 = []
for i in range(500):
    numbers = re.findall(r'-?\d+\.\d+', context1[i])
    if len(numbers) > 9:
        numbers = numbers[-9:]
    result_list1.append(float(numbers[1]))

    numbers = re.findall(r'-?\d+\.\d+', context2[i])
    if len(numbers) > 9:
        numbers = numbers[-9:]
    elif len(numbers) == 0:
        result_list2.append(result_list2[-1])
        continue
    result_list2.append(float(numbers[1]))
result_list2[28] = -6.4111e-04
result_list2[62] = 3.2723e-03
plt.plot(range(len(result_list1)), result_list1, label = "with abs")
plt.plot(range(len(result_list2)), result_list2, label = "without abs")
plt.legend()
plt.xlabel('Rounds')
plt.ylabel('value')
# plt.yticks(ticks)
plt.title('comparsion of scalling value')
plt.savefig('/home/jianhuiwei/rsch/jianhui/FedLoGe/ABS analysis100_2.png', dpi=600)

