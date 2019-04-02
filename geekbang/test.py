import pandas as pd
import numpy as np
import tensorflow as tf


def std(df):
    return df.apply(lambda column: (column - column.mean()) / column.std())


# 看一下能不能算出函数 y = 8x + 465
data1 = pd.read_csv('test.csv', names=['x_label', 'y_label'])
# df = std(data1)
df = data1
ones = pd.DataFrame({'ones': np.ones(len(df))})
df = pd.concat([ones, df], axis=1) # 根据列合并数据

# 数据处理

x_data = np.array(df[['ones', 'x_label']])
y_data = np.array(df[['y_label']])

# 创建线性回归模型
alpha = 0.0001
epoch = 1000000

x = tf.placeholder(tf.float32, x_data.shape)
y = tf.placeholder(tf.float32, y_data.shape)
w = tf.get_variable("weights", (x_data.shape[1], 1), initializer=tf.constant_initializer())
y_pred = tf.matmul(x, w)
loss_op = 1 / (2 * len(x_data)) * tf.matmul((y_pred - y), (y_pred - y), transpose_a=True)
opt = tf.train.GradientDescentOptimizer(learning_rate=alpha)
train_op = opt.minimize(loss_op)

with tf.Session() as sess:
    # 初始化全局变量
    sess.run(tf.global_variables_initializer())
    # 开始训练模型，
    # 因为训练集较小，所以每轮都使用全量数据训练
    for e in range(1, epoch + 1):
        sess.run(train_op, feed_dict={x: x_data, y: y_data})
        if e % 10000 == 0:
            loss, W = sess.run([loss_op, w], feed_dict={x: x_data, y: y_data})
            log_str = "Epoch %d \t Loss=%.3g \t Model: y = %.4gx1 + %.4gx2"
            print(log_str % (e, loss, W[1], W[0]))













