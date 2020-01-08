# -*- coding: utf-8 -*-

# @author: Awesome_Tang
# @date: 2018-12-16
# @version: python2.7

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class constant(object):
    classes = 10  # 类别数
    alpha = 0.01  # 学习率
    steps = 100  # 迭代次数
    batch_size = 50  # 每批次训练样本数
    print_per_batch = 10  # 每多少轮输出一次结果



class SoftMax():

    def __init__(self, constant):
        self.constant = constant
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.input_x = tf.placeholder(tf.float32, [None, 784], name='input_x')
        self.input_y = tf.placeholder(
            tf.float32, [None, self.constant.classes], name='input_y')

        self.run_model()

    def run_model(self):
        # define variables: weights and biases
        weight = tf.Variable(tf.zeros([784, 10]))
        bias = tf.Variable(tf.zeros([10]))

        # define model
        y = tf.nn.softmax(tf.matmul(self.input_x, weight) + bias)

        # define loss function
        cross_entropy = - tf.reduce_sum(self.input_y * tf.log(y))
        train_step = tf.train.GradientDescentOptimizer(
            self.constant.alpha).minimize(cross_entropy)
        correct_prediction = tf.equal(
            tf.argmax(y, 1), tf.argmax(self.input_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        # initial variables
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        t1 = time.time()
        for i in range(self.constant.steps):
            batch = self.mnist.train.next_batch(self.constant.batch_size)
            if i % self.constant.print_per_batch == 0:
                train_accuracy = accuracy.eval(session=sess,
                                               feed_dict={self.input_x: batch[0], self.input_y: batch[1]})
                print("step %d, train_accuracy %g" % (i, train_accuracy))
            train_step.run(session=sess, feed_dict={self.input_x: batch[0], self.input_y: batch[1]})

        t2 = time.time()
        print(t2-t1)
        print("test accuracy %g" % accuracy.eval(session=sess,
                                                 feed_dict={self.input_x: self.mnist.test.images, self.input_y: self.mnist.test.labels}))
        sess.close()

if __name__ == "__main__":
    SoftMax(constant)
