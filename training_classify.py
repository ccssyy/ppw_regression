import tensorflow as tf
import load_data
import vgg_inference_classify
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
import numpy as np
import time
import h5py
import matplotlib.pyplot as plt


def training(batch_size=256):
    X = tf.placeholder(dtype=tf.float32, shape=[None, 1, 500, 1], name='input')
    Y = tf.placeholder(dtype=tf.float32, shape=[None], name='output')

    train_X = load_data.get_trainset('train_set_v3.h5', 'ppw1').reshape([-1, 1, 500, 1])
    train_Y = load_data.get_trainset('train_set_v3.h5', 'label_nsbp_c').reshape([-1])
    test_X = load_data.get_testset('test_set_v3.h5', 'ppw1').reshape([-1, 1, 500, 1])
    test_Y = load_data.get_testset('test_set_v3 .h5', 'label_nsbp_c').reshape([-1])

    logits = vgg_inference_classify.inference_op(X, 0.5)
    # print(Y.get_shape())
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.to_int64(Y), logits=logits))
    # tf.add_to_collection('losses', mse_loss)
    # losses = tf.add_n(tf.get_collection('losses'))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)
    train_op = optimizer.minimize(cost)

    def accuracy(labels, output):
        labels = tf.to_int64(labels)
        predict_result = tf.equal(labels, tf.argmax(output, 1))
        acc = tf.reduce_mean(tf.cast(predict_result, tf.float32))
        return acc

    '''
    # 数据标准化处理
    ss_x = StandardScaler()
    x_train = ss_x.fit_transform(train_X)
    x_test = ss_x.fit_transform(test_X)

    ss_y = StandardScaler()
    y_train = ss_y.fit_transform(train_Y)
    y_train = y_train.reshape(-1)
    print(y_train.shape)
    y_test = ss_y.fit_transform(test_Y)
    y_test = y_test.reshape(-1)
    '''

    data_len = len(train_X)
    idx_array = np.arange(data_len)
    step = data_len // batch_size
    acc_op = accuracy(Y, logits)
    acc = {'epoch': [], 'acc': [], 'cost': []}

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        sess.run(init)

        for epoch in range(1, 1+1000):
            np.random.shuffle(idx_array)
            shuffle_train_X = train_X[idx_array]
            # print('shuffle_train_X:{}'.format(shuffle_train_X))
            shuffle_train_Y = train_Y[idx_array]
            # print('shuffle_train_Y:{}'.format(shuffle_train_Y))
            # shuffle_x_train = x_train[idx_array]
            # shuffle_y_train = y_train[idx_array]
            for i in range(step):
                start_time = time.time()
                start = i * batch_size
                end = start + batch_size
                feed_dict = {X: shuffle_train_X[start:end], Y: shuffle_train_Y[start:end]}
                # print('X:{}'.format(len(feed_dict[X])))
                _, loss, batch_acc, out = sess.run([train_op, cost, acc_op, logits], feed_dict=feed_dict)
                duration = time.time() - start_time
                print('Epoch {0} step {1}/{2}: loss:{3:.6f}, batch_acc:{4:.4f} ({5:.3f} sec/step)'.format(epoch, i+1, step,
                                                                                                          loss, batch_acc, duration))
            if epoch % 50 == 0:
                test_feed_dict = {X: test_X, Y: test_Y}
                predict_acc, test_loss, output = sess.run([acc_op, cost, logits], feed_dict=test_feed_dict)
                acc['acc'].append(predict_acc)
                acc['epoch'].append(epoch)
                acc['cost'].append(test_loss)
                print('test_output:{}'.format(output))
                print('loss in testset:{0:.6f},    accuracy:{1:.4f}'.format(test_loss, predict_acc))
                plt.title('Classify_Epoch-{0}'.format(epoch))


    print('Train Finished !!!')
    print(acc)


def main():
    training(1024)


if __name__ == '__main__':
    main()
