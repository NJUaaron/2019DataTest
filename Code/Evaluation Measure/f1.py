#计算文件夹内所有csv的f1
import os
import numpy as np
import pandas as pd
from sklearn import metrics

#calculate the rows of data
def file_len(fname):
    with open(fname) as f:
        for i in enumerate(f):
            pass
    return i + 1

path = r'模型预测出的中间结果路径'
#将该文件夹下的所有文件名存入一个列表
file_list = os.listdir(path)
cnt = 0
not_cnt = 0
acc_f1 = 0     #累积f1

print(file_list[::5])  #每5个取一个

#循环遍历列表中各个CSV文件名，每五个文件计算一个平均f1
for i in range(0, len(file_list)):
    file = path + file_list[i]
    df = pd.read_csv(file)
    bug = df.values[0:file_len(file)-1, 1]
    pre = df.values[0:file_len(file)-1, 2]

    thres = np.median(pre)#取中位数为阈值
    for k in range(0, len(pre)):
        if pre[k] > thres:
            pre[k] = 1
        else:
            pre[k] = 0

    #bug列全零或全1则不参与计算
    all_zero = 1  #是否全零
    for j in range(0, len(bug)):
        if bug[j] == 1:
            all_zero = 0
            break

    all_one = 1  #是否全1
    for j in range(0, len(bug)):
        if bug[j] == 0:
            all_one = 0
            break

    if all_zero == 1 or all_one == 1:
        not_cnt += 1
    else:
        f1 = metrics.f1_score(bug, pre)
        acc_f1 += f1

    cnt += 1
    if cnt >= 5:
        print(round(acc_f1/(cnt-not_cnt), 4))#保留4位小数
        acc_f1 = 0
        cnt = 0
        not_cnt = 0
