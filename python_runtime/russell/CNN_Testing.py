# ------------------------------------------------------------------
#
# CNN Testing
#
# ------------------------------------------------------------------


from __future__ import print_function

import os

# Set GPU number.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Import MNIST data
import input_data

fullsets = input_data.read_data_sets()

import tensorflow as tf
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

x = tf.placeholder(tf.float32, shape=[None, 300])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


# Handy functions
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# Convolution and Pooling:
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# First Convolution Layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

# apply ReLU function and max pool
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second Convolution Layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densely Connected Layer with 1024 neurons:
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout before the readout layer to reduce overfitting
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout layer:
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

# apply regularization
regularizer = tf.nn.l2_loss(W_fc2)
cross_entropy = tf.reduce_mean(cross_entropy + 0.01 * regularizer)

adam_optimizer = tf.train.AdamOptimizer(1e-4)

train_step = adam_optimizer.minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# for plotting
iteration_list = []
train_loss_list = []
test_accuracy_list = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(20000):
        batch = fullsets.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy));
            print("current learning rate is: " + str(adam_optimizer._lr))

            train_loss_list.append(
                cross_entropy.eval(feed_dict={x: fullsets.test.images, y_: fullsets.test.labels, keep_prob: 1.0}));

            test_accuracy_list.append(
                accuracy.eval(feed_dict={x: fullsets.test.images, y_: fullsets.test.labels, keep_prob: 1.0}));

            iteration_list.append(i);

        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: fullsets.test.images, y_: fullsets.test.labels, keep_prob: 1.0}))

    plt.ylabel('training loss')
    plt.xlabel('num of trainig epoch/iteration')
    plt.title('ConvNet Training on fullsets')
    plt.plot(iteration_list, train_loss_list, color='r', label='train loss')
    plt.legend(loc='upper right')
    plt.savefig('./Problem2_figure_Training_Loss.png')

    plt.gcf().clear()

    plt.ylabel('test accuracy')
    plt.xlabel('num of trainig epoch/iteration')
    plt.title('ConvNet Training on fullsets')
    plt.plot(iteration_list, test_accuracy_list, color='y', label='test accuracy')
    plt.legend(loc='lower right')
    plt.savefig('./Problem2_figure_Test_Acc.png')
