import tensorflow as tf
normal = tf.Variable("Elephant", tf.string)
ignitiong = tf.Variable(451, tf.int16)
floating = tf.Variable(3.1415, tf.float64)
its_complicated = tf.Variable(12.3 - 4.89j, tf.complex64)

# 1阶张量
mystr = tf.Variable(["Hello", "World"], tf.string)
cool_numbers  = tf.Variable([3.14159, 2.71828], tf.float32)
first_primes = tf.Variable([2, 3, 5, 7, 11], tf.int32)
its_very_complicated = tf.Variable([12.3 - 4.85j, 7.5 - 6.23j], tf.complex64)

# 2阶张量
mymat = tf.Variable([[7],[11]], tf.int16)
myxor = tf.Variable([[False, True],[True, False]], tf.bool)
linear_squares = tf.Variable([[4], [9], [16], [25]], tf.int32)
squarish_squares = tf.Variable([ [4, 9], [16, 25] ], tf.int32)
rank_of_squares = tf.rank(squarish_squares)
mymatC = tf.Variable([[7],[11]], tf.int32)

# 4阶张量
my_image = tf.zeros([10, 299, 299, 3])  # batch x height x width x colo