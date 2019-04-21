#单个文件进行1次5-fold，保存5次的预测结果
import tensorflow as tf
import numpy as np
import pandas as pd

sess = tf.Session()

#calculate the rows of data
def file_len(fname):
    with open(fname) as f:
        for i in enumerate(f):
            pass
    return i + 1

#read data from CSV, save in CSV
file = r"读取的csv文件"
resultfile = r"保存路径"    #会产生5个预测结果文件

df = pd.read_csv(file)

seed = tf.set_random_seed(1)
np.random.seed(seed)

metrix_num = 17     #metrix的数量

#将数据等分为5份
spart = int(file_len(file)/5)
part = [0, spart, 2*spart, 3*spart, 4*spart, file_len(file)-1]
auc_sum = 0

for index in range(5):
    #取一份作为测试集，其余作为训练集
    begin_part = part[index]
    end_part = part[index + 1]

    x_vals_test = df.values[begin_part:end_part, 0:metrix_num]
    y_vals_test = df.values[begin_part:end_part, metrix_num]
    loc_test = df.values[begin_part:end_part, 0]

    if index == 0:
        x_vals_train = df.values[end_part:part[5], 0:metrix_num]
        y_vals_train = df.values[end_part:part[5], metrix_num]
    elif index == 4:
        x_vals_train = df.values[0:begin_part, 0:metrix_num]
        y_vals_train = df.values[0:begin_part, metrix_num]
    else:
        x_vals_train_1 = df.values[0:begin_part, 0:metrix_num]
        y_vals_train_1 = df.values[0:begin_part, metrix_num]
        x_vals_train_2 = df.values[end_part:part[5], 0:metrix_num]
        y_vals_train_2 = df.values[end_part:part[5], metrix_num]
        x_vals_train = np.append(x_vals_train_1, x_vals_train_2, axis=0)
        y_vals_train = np.append(y_vals_train_1, y_vals_train_2, axis=0)


    #正则化
    def normalize_cols(m):
        col_max = m.max(axis=0)
        col_min = m.min(axis=0)
        return (m - col_min) / (col_max - col_min)


    x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
    x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))


    batch_size = 50
    x_data = tf.placeholder(shape=[None, metrix_num], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    #隐层节点数量
    hidden_layer_nodes1 = 15
    hidden_layer_nodes2 = 15

    A1 = tf.Variable(tf.random_normal(shape=[metrix_num, hidden_layer_nodes1]))
    b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes1]))  #hidden layer1
    A2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes1, hidden_layer_nodes2]))
    b2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes2]))  #hidden layer2
    A3 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes2, 1]))
    b3 = tf.Variable(tf.random_normal(shape=[1]))                   #output layer

    hiddden_output1 = tf.add(tf.matmul(x_data, A1), b1)
    hiddden_output2 = tf.add(tf.matmul(hiddden_output1, A2), b2)
    final_output = tf.nn.sigmoid(tf.add(tf.matmul(hiddden_output2, A3), b3))

    #损失函数
    loss = -tf.reduce_mean(y_target * tf.log(final_output + 1e-10) + (1 - y_target) * tf.log(1 - final_output + 1e-10))

    #步长
    my_opt = tf.train.AdamOptimizer(0.00005)
    train_step = my_opt.minimize(loss)

    sess.run(tf.global_variables_initializer())

    #开始训练
    for i in range(10000):
        #First we select a random set of indices for the batch
        rand_index = np.random.choice(len(x_vals_train), size=batch_size)
        #Then we select the training valuses
        rand_x = x_vals_train[rand_index]
        rand_y = np.transpose([y_vals_train[rand_index]])
        #now we run the training step
        sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    test_output = sess.run(final_output, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
    test_output_vec = np.transpose(test_output)[0]
    result_save = open(resultfile + str(index) + '.csv', 'w')
    result_save.write('loc,bug,pre\n')
    for i in range(len(test_output_vec)):
        result_save.write(str(loc_test[i]) + ',' + str(y_vals_test[i]) + ',' + str(test_output_vec[i]) + '\n')
    result_save.close()
    print('result saved.')

print('Output completed')







