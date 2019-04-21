#计算文件夹内所有csv的ce
import os
import pandas as pd
import numpy as np

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
acc_ce = 0     #累积ce
cutoff = 0.2

print(file_list[::5])  #每5个取一个

#循环遍历列表中各个CSV文件名，每五个文件计算一个平均ce
for i in range(0, len(file_list)):
    file = path + file_list[i]
    df = pd.read_csv(file)

    df = df.sort_values('pre', ascending=False)#按pre降序排序

    leng = file_len(file) - 1
    loc = df.values[0:leng, 0]
    bug = df.values[0:leng, 1]

    cumXs = np.cumsum(loc)  # x: LOC%
    cumYs = np.cumsum(bug)  # y: Bug%

    if cumYs[leng - 1] == 0:
        not_cnt += 1
    else:
        Xs = cumXs/cumXs[leng - 1]
        Ys = cumYs/cumYs[leng - 1]

        fix_subareas = [0] * leng
        fix_subareas[0] = 0.5 * Ys[0] * Xs[0]
        fix_subareas[1:leng] = 0.5 * (Ys[0:(leng-1)] + Ys[1:leng]) * abs(Xs[0:(leng-1)] - Xs[1:leng])

        pos = int(cutoff * leng)
        Xpos = Xs[pos-1]  # x-axis
        Ypos = Ys[pos-1]
        subareas = [0] * pos
        if pos == 1:
            subareas[0] = 0.5 * Xpos * Ypos
        elif pos == 2:
            subareas[0] = fix_subareas[0]
            subareas[1] = (Ypos+Ys[0])*(abs(Xpos-Xs[0]))*0.5
        else:
            subareas[0:(pos-1)] = fix_subareas[0:(pos-1)]
            subareas[pos-1] = (Ypos+Ys[pos-2])*(abs(Xpos-Xs[pos-2]))*0.5

        ce = np.sum(subareas)
        acc_ce += ce

    cnt += 1
    if cnt >= 5:
        print(round(acc_ce/(cnt-not_cnt), 4))#保留4位小数
        acc_ce = 0
        cnt = 0
        not_cnt = 0
