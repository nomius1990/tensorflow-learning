import tensorflow as tf
# 定义常量操作
hello = tf.constant("Hello Tensorflow")
# 创建一个会话
sess = tf.Session()
# 执行常量操作 hello 并且打印到标准输出
print(sess.run(hello))