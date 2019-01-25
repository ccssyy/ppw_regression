import tensorflow as tf
import numpy as np


# 定义卷积操作
def conv_op(input_op, name, kh, kw, n_out, sh, sw):
    input_op = tf.convert_to_tensor(input_op)
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "W", shape=[kh, kw, n_in, n_out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        # tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0.01)(kernel))
        conv = tf.nn.conv2d(input_op, kernel, [1, sh, sw, 1], padding='SAME')
        bias_init_val = tf.constant(0.1, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='B')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
    return activation


# 定义全连接操作
def fc_op(input_op, name, n_out):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "W",
                                 shape=[n_in, n_out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        # tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0.01)(kernel))
        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name='B')
        # tf.nn.rely_layer对输入变量input_op与kernel做矩阵乘法加上bias，在做RELU非线性变换得到activation
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
    return activation


# 定义池化层
def mpool_op(input_op, name, kh, kw, sh, sw):
    return tf.nn.max_pool(input_op,
                      ksize=[1, kh, kw, 1],
                      strides=[1, sh, sw, 1],
                      padding='SAME',
                      name=name)


def inference_op(input_op, keep_prob):
    # block 1
    conv1_1 = conv_op(input_op, name='conv1_1', kh=1, kw=3, n_out=16, sh=1, sw=1)
    conv1_2 = conv_op(conv1_1, name='conv1_2', kh=1, kw=3, n_out=16, sh=1, sw=1)
    pool1 = mpool_op(conv1_2, name='pool1', kh=1, kw=2, sh=1, sw=2)

    # block 2
    conv2_1 = conv_op(pool1, name='conv2_1', kh=1, kw=3, n_out=32, sh=1, sw=1)
    conv2_2 = conv_op(conv2_1, name='conv2_2', kh=1, kw=3, n_out=32, sh=1, sw=1)
    pool2 = mpool_op(conv2_2, name='pool2', kh=1, kw=2, sh=1, sw=2)

    # block 3
    conv3_1 = conv_op(pool2, name='conv3_1', kh=1, kw=3, n_out=64, sh=1, sw=1)
    conv3_2 = conv_op(conv3_1, name='conv3_2', kh=1, kw=3, n_out=64, sh=1, sw=1)
    conv3_3 = conv_op(conv3_2, name='conv3_3', kh=1, kw=3, n_out=64, sh=1, sw=1)
    pool3 = mpool_op(conv3_3, name='pool3', kh=1, kw=2, sh=1, sw=2)

    # block 4
    conv4_1 = conv_op(pool3, name='conv4_1', kh=1, kw=3, n_out=128, sh=1, sw=1)
    conv4_2 = conv_op(conv4_1, name='conv4_2', kh=1, kw=3, n_out=128, sh=1, sw=1)
    conv4_3 = conv_op(conv4_2, name='conv4_3', kh=1, kw=3, n_out=128, sh=1, sw=1)
    pool4 = mpool_op(conv4_3, name='pool4', kh=1, kw=2, sh=1, sw=2)

    # block 5
    conv5_1 = conv_op(pool4, name='conv5_1', kh=1, kw=3, n_out=128, sh=1, sw=1)
    conv5_2 = conv_op(conv5_1, name='conv5_2', kh=1, kw=3, n_out=128, sh=1, sw=1)
    conv5_3 = conv_op(conv5_2, name='conv5_3', kh=1, kw=3, n_out=128, sh=1, sw=1)
    pool5 = mpool_op(conv5_3, name='conv5_3', kh=1, kw=2, sh=1, sw=2)

    # flatten
    shp = pool5.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool5, [-1, flattened_shape], name='resh1')

    # fully connected
    fc6 = fc_op(resh1, name='fc6', n_out=1024)
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name="fc6_drop")

    fc7 = fc_op(fc6_drop, name="fc7", n_out=1024)
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_drop")

    logits = fc_op(fc7_drop, name='fc8', n_out=2)

    return logits


