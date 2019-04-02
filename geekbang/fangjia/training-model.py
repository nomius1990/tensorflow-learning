# 房价预测模型

import pandas as pd
import numpy as np
import tensorflow as tf


# 标准化,等比缩小  x1 = (x - x的平均值) / x的标准差
def normalize_feature(df):
    return df.apply(lambda column: (column - column.mean()) / column.std())


df = pd.read_csv('data1.csv', names=['square', 'bedrooms', 'price'])
df = normalize_feature(df)

ones = pd.DataFrame({'ones': np.ones(len(df))})
df = pd.concat([ones, df], axis=1)
aa = df.head()


# 数据处理: 获取x and y

X_data = np.array(df[['ones', 'square', 'bedrooms']])  # X_data = np.array(df.columns[0:3])
y_data = np.array(df[['price']])  # y_data = np.array(df[df.columns[-1]]).reshape(len(df),1)

# print(X_data.shape, type(X_data))
# print(y_data.shape, type(y_data))

# 创建线性回归模型(数据流图)
alpha = 0.01
epoch = 500
X = tf.placeholder(tf.float32, X_data.shape)  # 输入 X，形状[47, 3]
y = tf.placeholder(tf.float32, y_data.shape)  # 输出 y，形状[47, 1]

# 权重变量 W，形状[3,1]
W = tf.get_variable("weights", (X_data.shape[1], 1), initializer=tf.constant_initializer())
y_pred = tf.matmul(X, W)

loss_op = 1 / (2 * len(X_data)) * tf.matmul((y_pred - y), (y_pred - y), transpose_a=True)

opt = tf.train.GradientDescentOptimizer(learning_rate=alpha)
train_op = opt.minimize(loss_op)

# 创建会话(运行环境)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for e in range(1, epoch + 1):
        sess.run(train_op, feed_dict={X: X_data, y: y_data})
        if e % 10 == 0:
            loss, w = sess.run([loss_op, W], feed_dict={X: X_data, y: y_data})
            log_str = "Epoch %d \t Loss=%.3g \t Model: y = %.4gx1 + %.4gx2 + %.4g"
            print(log_str % (e, loss, w[1], w[2], w[0]))


























