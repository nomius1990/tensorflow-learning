import math
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
california_housing_dataframe = pd.read_csv("tensorflow/california_housing_train.csv", sep=",")
# 打乱并且进行水机排序这样不回顺怀随机梯度下降法的效果
california_housing_dataframe = california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe["median_house_value"] /= 1000.0
# california_housing_dataframe
des = california_housing_dataframe.describe() # 信息摘要 样本数,均值,标准偏差 最大值 最小值和各种分位数

# 在本练习中,我们将尝试预测 median_house_value 它将是我们的标签,我们将使用 total_rooms 作为输入特征
# 我们使用的是城市街区级别的数据,因此该特征表示相应街区的房间总数
# 为了训练模型,我们将使用tf estimator api 提供的 LinearRegressor 接口. 此Api 负责处理大量低级别模型搭建工作.
# 并且会提供执行模型训练,评估和推理的便利方法

# 第一步 定义特征兵配置特征列
# 为了将我们训练数据导入tf 我们需要指定每个特征包含的数据类型.在本练和今后的联系中 我们主要使用一下两类数据
# (1)分类数据: 一种文字数据.在本练习中.我们的住房数据集不包含任何分类特征,但您可能会看到示例包含家居风格和房地产广告词

my_feature = california_housing_dataframe[["total_rooms"]]
feature_columns = [tf.feature_column.numeric_column("total_rooms")]

# 第二步 定义目标 median_house_value
targets = california_housing_dataframe["median_house_value"]

# 第三步 配置 LinearRegressor
# 接下来，我们将使用 LinearRegressor 配置线性回归模型，
# 并使用 GradientDescentOptimizer（它会实现小批量随机梯度下降法 (SGD)）训练该模型。
# learning_rate 参数可控制梯度步长的大小。
# 注意：为了安全起见，我们还会通过 clip_gradients_by_norm 将梯度裁剪应用到我们的优化器。
# 梯度裁剪可确保梯度大小在训练期间不会变得过大，梯度过大会导致梯度下降法失败。

my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=my_optimizer
)

# 第四步 定义输入函数
# 要将加利福尼亚州住房数据导入 LinearRegressor，我们需要定义一个输入函数，让它告诉 TensorFlow 如何对数据进行预处理，
# 以及在模型训练期间如何批处理、随机处理和重复数据。
#
# 首先，我们将 Pandas 特征数据转换成 NumPy 数组字典。然后，
# 我们可以使用 TensorFlow Dataset API 根据我们的数据构建 Dataset 对象，
# 并将数据拆分成大小为 batch_size 的多批数据，
# 以按照指定周期数 (num_epochs) 进行重复。
#
# 注意：如果将默认值 num_epochs=None 传递到 repeat()，输入数据会无限期重复。
# 然后，如果 shuffle 设置为 True，则我们会对数据进行随机处理，以便数据在训练期间以随机方式传递到模型。
# buffer_size 参数会指定 shuffle 将从中随机抽样的数据集的大小。
# 最后，输入函数会为该数据集构建一个迭代器，并向 LinearRegressor 返回下一批数据。


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):

    """Trains a linear regression model of one feature.

    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """

    # Convert pandas data into a dict of np arrays.
    features = {key: np.array(value) for key, value in dict(features).items()}

    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features, targets))  # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


# 训练模型
# _ = linear_regressor.train(
#     input_fn=lambda: my_input_fn(my_feature, targets),
#     steps=100
# )

# 评估模型
# Create an input function for predictions.
# Note: Since we're making just one prediction for each example, we don't
# need to repeat or shuffle the data here.
# prediction_input_fn = lambda: my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)

# Call predict() on the linear_regressor to make predictions.
# predictions = linear_regressor.predict(input_fn=prediction_input_fn)
#
# # Format predictions as a NumPy array, so we can calculate error metrics.
# predictions = np.array([item['predictions'][0] for item in predictions])
#
# # Print Mean Squared Error and Root Mean Squared Error.
# mean_squared_error = metrics.mean_squared_error(predictions, targets)
# root_mean_squared_error = math.sqrt(mean_squared_error)
# print("Mean Squared Error (on training data): %0.3f" % mean_squared_error)
# print("Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error)
#
# min_house_value = california_housing_dataframe["median_house_value"].min()
# max_house_value = california_housing_dataframe["median_house_value"].max()
# min_max_difference = max_house_value - min_house_value
#
# print("Min. Median House Value: %0.3f" % min_house_value)
# print("Max. Median House Value: %0.3f" % max_house_value)
# print("Difference between Min. and Max.: %0.3f" % min_max_difference)
# print("Root Mean Squared Error: %0.3f" % root_mean_squared_error)


# calibration_data = pd.DataFrame()
# calibration_data["predictions"] = pd.Series(predictions)
# calibration_data["targets"] = pd.Series(targets)
# newdes = calibration_data.describe()
#
# sample = california_housing_dataframe.sample(n=300)
# x_0 = sample["total_rooms"].min()
# x_1 = sample["total_rooms"].max()
#
# # Retrieve the final weight and bias generated during training.
# weight = linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]
# bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')
#
# # Get the predicted median_house_values for the min and max total_rooms values.
# y_0 = weight * x_0 + bias
# y_1 = weight * x_1 + bias
#
# # Plot our regression line from (x_0, y_0) to (x_1, y_1).
# plt.plot([x_0, x_1], [y_0, y_1], c='r')
#
# # Label the graph axes.
# plt.ylabel("median_house_value")
# plt.xlabel("total_rooms")
#
# # Plot a scatter plot from our data sample.
# plt.scatter(sample["total_rooms"], sample["median_house_value"])
#
# # Display graph.
# plt.show()


def train_model(learning_rate, steps, batch_size, input_feature="total_rooms"):
    """Trains a linear regression model of one feature.

    Args:
      learning_rate: A `float`, the learning rate.
      steps: A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
      batch_size: A non-zero `int`, the batch size.
      input_feature: A `string` specifying a column from `california_housing_dataframe`
        to use as input feature.
    """

    periods = 10
    steps_per_period = steps / periods

    my_feature = input_feature
    my_feature_data = california_housing_dataframe[[my_feature]]
    my_label = "median_house_value"
    targets = california_housing_dataframe[my_label]

    # Create feature columns.
    feature_columns = [tf.feature_column.numeric_column(my_feature)]

    # Create input functions.
    training_input_fn = lambda: my_input_fn(my_feature_data, targets, batch_size=batch_size)
    prediction_input_fn = lambda: my_input_fn(my_feature_data, targets, num_epochs=1, shuffle=False)

    # Create a linear regressor object.
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=feature_columns,
        optimizer=my_optimizer
    )

    # Set up to plot the state of our model's line each period.
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.title("Learned Line by Period")
    plt.ylabel(my_label)
    plt.xlabel(my_feature)
    sample = california_housing_dataframe.sample(n=300)
    plt.scatter(sample[my_feature], sample[my_label])
    colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("RMSE (on training data):")
    root_mean_squared_errors = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        linear_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )
        # Take a break and compute predictions.
        predictions = linear_regressor.predict(input_fn=prediction_input_fn)
        predictions = np.array([item['predictions'][0] for item in predictions])

        # Compute loss.
        root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(predictions, targets))
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, root_mean_squared_error))
        # Add the loss metrics from this period to our list.
        root_mean_squared_errors.append(root_mean_squared_error)
        # Finally, track the weights and biases over time.
        # Apply some math to ensure that the data and line are plotted neatly.
        y_extents = np.array([0, sample[my_label].max()])

        weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % input_feature)[0]
        bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

        x_extents = (y_extents - bias) / weight
        x_extents = np.maximum(np.minimum(x_extents,
                                          sample[my_feature].max()),
                               sample[my_feature].min())
        y_extents = weight * x_extents + bias
        plt.plot(x_extents, y_extents, color=colors[period])
    print("Model training finished.")

    # Output a graph of loss metrics over periods.
    plt.subplot(1, 2, 2)
    plt.ylabel('RMSE')
    plt.xlabel('Periods')
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(root_mean_squared_errors)

    # Output a table with calibration data.
    calibration_data = pd.DataFrame()
    calibration_data["predictions"] = pd.Series(predictions)
    calibration_data["targets"] = pd.Series(targets)
    display.display(calibration_data.describe())

    print("Final RMSE (on training data): %0.2f" % root_mean_squared_error)


train_model(
    learning_rate=0.00002,
    steps=1000,
    batch_size=5,
    input_feature="population"
)