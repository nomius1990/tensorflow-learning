import tensorflow as tf

# 常量操作
a = tf.constant(2)
b = tf.constant(3)

#常量在初始化的时候就赋值进去了
with tf.Session() as sess:
    print("i get a is %i" % sess.run(a))
    print("i get b is %i" % sess.run(b))
    print("i get a + b  is %i" % sess.run(a + b))
    print("i get a * b  is %i" % sess.run(a * b))


# 占位符操作
x = tf.placeholder(tf.int16, shape=(), name="x")
y = tf.placeholder(tf.int16, shape=(), name="y")

add = tf.add(x, y)
mul = tf.multiply(x, y)
with tf.Session() as sess:
    print(sess.run(add, feed_dict={x: 2, y: 3}))  # feed_dict 从字典中获取值
    print(sess.run(mul, feed_dict={x: 2, y: 3}))
