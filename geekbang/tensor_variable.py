# 变量是一种特殊的张量
# 变量的主要作用是维护特定节点的状态,如深度学习或机器学习的模型参数
import tensorflow as tf

# 创建变量
# tf.random_normal 方法返回形状为(1，4)的张量。它的4个元素符合均值为100、标准差为0.35的正态分布。
W = tf.Variable(initial_value=tf.random_normal(shape=(1, 4), mean=100, stddev=0.35), name="W")
b = tf.Variable(tf.zeros([4]), name="b")

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run([W, b])  # run 的时候才会执行实际的操作

sess.run(tf.assign_add(b, [1, 1, 1, 1]))
sess.run(b)

filename = './summary/test.ckpt'
saver = tf.train.Saver({'W': W, 'b': b})
saver.save(sess, filename, global_step=0)  # 执行save 的时候才会保存到文件
