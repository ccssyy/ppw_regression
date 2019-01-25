import tensorflow as tf
import load_data
import vgg_inference
from sklearn.preprocessing import StandardScaler
import numpy as np
import time
import h5py
import os
import scipy.io as sio
import matplotlib.pyplot as plt


def get_segment_average(mat_file):
    data = sio.loadmat(mat_file)
    average = []
    segment = 1
    segment_sum = 0
    count = 0
    for i in range(len(data['idx_segment'])):
        if data['idx_segment'][i][0] == segment:
            count += 1
            # a = data['nsbp'][i][0]
            segment_sum += int(data['nsbp'][i][0])
        else:
            average.append(segment_sum // count)
            count = 1
            segment = int(data['idx_segment'][i][0])
            segment_sum = int(data['nsbp'][i][0])
    average.append(segment_sum // count)

    segment_average = np.asarray(data['nsbp'], dtype=np.int32)
    # print(segment_average.shape)
    for i in range(len(segment_average)):
        idx = data['idx_segment'][i][0] - 1
        segment_average[i][0] = average[idx]
    return segment_average


def training(batch_size=256, lr=0.0001, kprob=0.8):
    dir_path = 'VGG16-16-32-64-128-128-1024-k-3-p-2-lr-00001_regularization-05_predict_correct_smoothly_v30'

    X = tf.placeholder(dtype=tf.float32, shape=[None, 1, 500, 1], name='input')
    Y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='output')
    # Y_segment_idx = tf.placeholder(dtype=tf.int32, shape=[batch_size], name='Y_segment_idx')

    predict_sbp = vgg_inference.inference_op(X, kprob)

    def loss_function(output, labels, depth):
        mse_loss = tf.losses.mean_squared_error(output, Y)
        max_ind = tf.argmax(output, 0)
        max_ind_onehot = tf.one_hot(max_ind, depth=depth, on_value=1, off_value=0)
        b_loss = tf.reduce_sum(tf.matmul(tf.to_float(max_ind_onehot), tf.pow(tf.subtract(output, labels), 2)))
        cost = mse_loss + 0 * b_loss
        tf.add_to_collection('losses', cost)
        losses = tf.add_n(tf.get_collection('losses'))
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_op = optimizer.minimize(losses)

        return train_op, mse_loss, b_loss, cost

    train_X = load_data.get_trainset('train_set_v8_random_3.h5', 'ppw1').reshape([-1, 1, 500, 1])
    train_Y = load_data.get_trainset('train_set_v8_random_3.h5', 'label_nsbp_r')
    # train_Y_segment_idx = load_data.get_trainset('train_set_v8_random_3.h5', 'segment').reshape(-1)
    # train_average_correct = load_data.get_trainset('train_set_v6.h5', 'segment_average')
    # added_train_X = np.hstack((train_X, train_average_correct)).reshape([-1, 1, 501, 1])
    train_Y = train_Y.reshape([-1, 1])
    test_X = load_data.get_testset('test_set_v8_random_3.h5', 'ppw1').reshape([-1, 1, 500, 1])
    test_Y = load_data.get_testset('test_set_v8_random_3.h5', 'label_nsbp_r')
    test_Y_segment_idx = load_data.get_testset('test_set_v8_random_3.h5', 'segment').reshape(-1)
    test_Y_segment_idx = np.array(test_Y_segment_idx, dtype=np.int32)
    idx_testdata = load_data.get_testset('test_set_v8_random_3.h5', 'testset_idx')
    # test_average_correct = load_data.get_testset('test_set_v6.h5', 'segment_average')
    # added_test_X = np.hstack((test_X, test_average_correct)).reshape([-1, 1, 501, 1])
    test_Y = test_Y.reshape([-1, 1])

    # print(added_train_X)
    # exit(0)

    train_op, mse_loss, b_loss, cost = loss_function(predict_sbp, Y, batch_size)
    _, test_mse_loss, _, test_cost = loss_function(predict_sbp, Y, test_Y.shape[0])

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

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True 
        sess.run(init)

        for epoch in range(1, 1 + 1000):
            np.random.shuffle(idx_array)
            shuffle_train_X = train_X[idx_array]
            # print('shuffle_train:{}'.format())
            shuffle_train_Y = train_Y[idx_array]
            # shuffle_x_train = x_train[idx_array]
            # shuffle_y_train = y_train[idx_array]
            for i in range(step):
                start_time = time.time()
                start = i * batch_size
                end = start + batch_size
                feed_dict = {X: shuffle_train_X[start:end], Y: shuffle_train_Y[start:end]}
                # print('X:{}'.format(len(feed_dict[X])))
                _, loss, mse_cost = sess.run([train_op, cost, mse_loss], feed_dict=feed_dict)
                duration = time.time() - start_time
                print(
                    'Epoch {0} step {1}/{2}: train_loss:{3:.6f}  mse:{4:.6f} ({5:.3f} sec/step)'.format(epoch, i, step,
                                                                                                        loss, mse_cost,
                                                                                                        duration))

            if epoch % 50 == 0:
                predict, total_loss, mse = sess.run([predict_sbp, test_cost, test_mse_loss],
                                                    feed_dict={X: test_X, Y: test_Y})
                print('mse_loss in testset:{0:.6f}\ntotal_loss:{1:.6f}'.format(mse, total_loss))
                dict = {}           # 每个段号下所有血压值
                for i in range(len(predict)):
                    if test_Y_segment_idx[i] not in dict.keys():
                        dict[test_Y_segment_idx[i]] = predict[i]
                    else:
                        dict[test_Y_segment_idx[i]] = np.append(dict[test_Y_segment_idx[i]], predict[i])
                # print(dict)
                for key in dict.keys():
                    # print('key:{}'.format(key))
                    if len(dict[key]) <= 2:
                        dict[key] = np.mean(dict[key], dtype=np.int32)
                    else:
                        argmax = np.argmax(dict[key])
                        dict[key] = np.delete(dict[key], argmax, axis=0)
                        argmin = np.argmin(dict[key])
                        dict[key] = np.delete(dict[key], argmin, axis=0)
                        dict[key] = np.mean(dict[key], dtype=np.int32)

                for i in range(len(predict)):
                    segment_id = test_Y_segment_idx[i]
                    # print(segment_id)
                    predict[i] = dict[segment_id]
                # print('predict:{}'.format(predict))
                # exit(0)
                if os.path.exists(dir_path) == False:
                    os.makedirs(dir_path)
                np.savetxt(dir_path + '/predict_test_epoch-'+str(epoch)+'.txt', predict, fmt='%d')
                correct_mse = np.mean(np.square(predict - test_Y))
                # correct_rmse = np.sqrt(correct_mse)
                rmse = np.sqrt(correct_mse)
                std = np.std(predict - test_Y)
                me = np.mean(predict - test_Y)
                mae = np.mean(np.abs(predict - test_Y))
                print('correct_MSE:{4:.6f}\nRMSE:{0:.6f}\nStd:{1:.6f}\nME:{2:.6f}\nMAE:{3:.6f}\n'.format(rmse, std, me,
                                                                                                         mae,
                                                                                                         correct_mse))

                # plt.figure(figsize=(500, 200))
                plt.plot(np.arange(len(test_Y)), test_Y, 'b-', np.arange(len(predict)),
                         predict, 'g-')
                plt.title('Epoch-' + str(epoch))
                plt.text(100, 150,
                         'lr={0},keepprob={1},batchsize={2},t_epoch={3}\nmse_loss:{4}\nRMSE:{5:.6f}\nStd:{6:.6f}\nME:{7:.6f}\nMAE:{8:.6f}\ntotal_loss:{9} \n'.format(
                             lr, kprob, batch_size, 1000, correct_mse, rmse, std, me, mae, total_loss))
                plt.savefig(dir_path + '/epoch-' + str(epoch))
                plt.close()
                if epoch == 1000:
                    with open(dir_path + '/result.txt', 'w', encoding='utf-8') as f:
                        f.writelines('index\tidx_data\ttest_label\tpredict\n')
                        for i in range(len(predict)):
                            f.writelines('{0}\t{1}\t{2}\t{3:.1f}\n'.format(i + 1, int(idx_testdata[i].tolist()+1),
                                                                           test_Y[i][0].tolist(),
                                                                           predict[i][0].tolist()))
                        f.writelines('MSE:{0:.6f}\n'.format(correct_mse))
                        f.writelines('RMSE:{0:.6f}\n'.format(rmse))
                        f.writelines('Std:{0:.6f}\n'.format(std))
                        f.writelines('ME:{0:.6f}\n'.format(me))
                        f.writelines('MAE:{0:.6f}\n'.format(mae))

    print('Train Finished !!!')


def main():
    lr = 0.0001
    keepprob = 0.8
    training(1024, lr, keepprob)


if __name__ == '__main__':
    main()
    # a = get_segment_average('alldata_v2.mat')
    # print(a)
