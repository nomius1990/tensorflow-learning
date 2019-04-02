# 会话提供了估算张量和执行操作的运行环境,他是发放计算任务的客户端,所有计算任务都由它连接的
# 执行引擎完成.一个会话的典型使用流程如下
# 1.获取张量
# 2.估算张量 / 执行操作
# 3.执行会话
import tensorflow as tf
# 创建数据流图：y = W * x + b，其中W和b为存储节点，x为数据节点。
x = tf.placeholder(tf.float32)
W = tf.Variable(1.0)
b = tf.Variable(1.0)
y = W * x + b
with tf.Session() as sess:
    tf.global_variables_initializer().run() # Operation.run
    fetch = y.eval(feed_dict={x: 3.0})      # Tensor.eval
    print(fetch)                            # fetch = 1.0 * 3.0 + 1.0