# -*- coding: utf-8 -*-

# @author: Awesome_Tang
# @date: 2018-12-15
# @version: python2.7

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class constant(object):
    """
    CNN 模型参数
    """
    classes = 10  # 类别数
    alpha = 1e-4  # 学习率
    keep_prob = 0.5  # 保留比例
    steps = 100  # 迭代次数
    batch_size = 50  # 每批次训练样本数
    tensorboard_dir = 'tensorboard/CNN'  # log输出路径
    print_per_batch = 10  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard


class CNN():

    def __init__(self, constant):
        self.constant = constant
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.input_x = tf.placeholder(tf.float32, [None, 784], name='input_x')
        self.input_y = tf.placeholder(
            tf.float32, [None, self.constant.classes], name='input_y')

        self.CNN_model()

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    def CNN_model(self):
        #  第一层： 卷积
        x_image = tf.reshape(self.input_x, [-1, 28, 28, 1])
        w_cv1 = self.weight_variable([3, 3, 1, 32])
        b_cv1 = self.bias_variable([32])
        h_cv1 = tf.nn.relu(self.conv2d(x_image, w_cv1) + b_cv1)
        h_mp1 = self.max_pool_2x2(h_cv1)

        # 第二层： 卷积
        w_cv2 = self.weight_variable([3, 3, 32, 64])
        b_cv2 = self.bias_variable([64])
        h_cv2 = tf.nn.relu(self.conv2d(h_mp1, w_cv2) + b_cv2)
        h_mp2 = self.max_pool_2x2(h_cv2)

        # 第三层： 全连接
        W_fc1 = self.weight_variable([7*7*64, 128])
        b_fc1 = self.bias_variable([128])

        h_mp2_flat = tf.reshape(h_mp2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_mp2_flat, W_fc1) + b_fc1)

        # 第四层： Dropout层
        keep_prob = tf.placeholder("float")
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # 第五层： softmax输出层
        W_fc2 = self.weight_variable([128, 10])
        b_fc2 = self.bias_variable([10])
        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        # 优化器&损失函数
        cross_entropy = -tf.reduce_sum(self.input_y * tf.log(y_conv))
        train_step = tf.train.AdamOptimizer(
            self.constant.alpha).minimize(cross_entropy)
        correct_prediction = tf.equal(
            tf.argmax(y_conv, 1), tf.argmax(self.input_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        loss = tf.reduce_mean(cross_entropy)
        
        # tensorboard配置
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("accuracy", accuracy)
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(self.constant.tensorboard_dir)

        # 初始化变量
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        t1 = time.time()
        for i in range(self.constant.steps):
            batch = self.mnist.train.next_batch(
                self.constant.batch_size)
            if i % self.constant.print_per_batch == 0:
                train_accuracy = accuracy.eval(session=sess,
                                               feed_dict={self.input_x: batch[0], self.input_y: batch[1], keep_prob: 1.0})
                print("step %d, train_accuracy %g" % (i, train_accuracy))
            train_step.run(session=sess, feed_dict={self.input_x: batch[0], self.input_y: batch[1],
                                                    keep_prob: self.constant.keep_prob})

            if i % self.constant.save_per_batch == 0:
                s = sess.run(merged_summary, feed_dict={
                             self.input_x: batch[0], self.input_y: batch[1], keep_prob: 1.0})
                writer.add_summary(s, i)

        t2 = time.time()
        print(t2-t1)
        print("test accuracy %g" % accuracy.eval(session=sess,
                                                 feed_dict={self.input_x: self.mnist.test.images, self.input_y: self.mnist.test.labels,
                                                            keep_prob: 1.0}))
        sess.close()

if __name__ == "__main__":
    CNN(constant)
