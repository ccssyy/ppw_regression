import tensorflow as tf
import load_data
import vgg_inference_3
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


def generate_mask(a_label, b_label):
    return tf.equal(tf.expand_dims(a_label, 1), tf.expand_dims(b_label, 0))


def training(batch_size=256, lr=0.0001, kprob=0.8):
    dir_path = 'VGG16-16-32-64-128-128-1024-k-3-p-2_lr-'+str(lr)+'_regularization-1e0_random-5_with_segment_loss_correct_predict_smoothly_v38'

    X = tf.placeholder(dtype=tf.float32, shape=(None, 1, 501, 1), name='input')
    Y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='label')
    segment_label = tf.placeholder(dtype=tf.int32, shape=(None), name='segment_label')

    predict_sbp, segment_benchmack = vgg_inference_3.inference_op(X, kprob)

    '''
    segment_mask = generate_mask(np.arange(2885, dtype=np.int32), segment_label)
    segment_predict = tf.multiply(tf.to_float(segment_mask), tf.reshape(predict_sbp, [-1]))
    segment_sum = tf.reduce_sum(segment_predict, axis=1)
    segment_predict_nonzero_count = tf.count_nonzero(segment_predict, axis=1, dtype=tf.float32)

    segment_benchmack = tf.divide(segment_sum, segment_predict_nonzero_count+0.00001)
    '''

    batch_segment_mask = generate_mask(np.arange(2885, dtype=np.int32), segment_label - 1)
    batch_segment_predict = tf.multiply(tf.to_float(batch_segment_mask), tf.reshape(predict_sbp, [-1]))
    batch_segment_sum = tf.reduce_sum(batch_segment_predict, axis=1)
    batch_segment_predict_nonzero_count = tf.count_nonzero(batch_segment_predict, axis=1, dtype=tf.float32)

    batch_segment_benchmack = tf.divide(batch_segment_sum, batch_segment_predict_nonzero_count + 0.00001)
    zero_mask_1 = tf.equal(segment_benchmack, 0.0)
    zero_mask_2 = tf.equal(batch_segment_benchmack, 0.0)
    two_zero_mask = tf.stack([zero_mask_1, zero_mask_2])
    zero_mask = tf.to_float(tf.reduce_any(two_zero_mask, axis=0))
    twotimes_zeromask = zero_mask * 2
    # print('zero_mask:{}'.format(zero_mask))
    # batch_segment_benchmack = batch_segment_benchmack + tf.multiply(batch_segment_benchmack, twotimes_zeromask)
    segment_benchmack = tf.assign(segment_benchmack,
                                  0.5 * (tf.multiply(segment_benchmack, twotimes_zeromask) + tf.multiply(batch_segment_benchmack, twotimes_zeromask)))

    # print(segment_benchmack)
    # exit(1)
    '''
    for i in range(5):
        # segment_benchmack = tf.get_variable(name='segment_benchmack')
        condition = tf.cond(
            tf.reshape(tf.equal(segment_benchmack[segment_label[i] - 1], tf.zeros(shape=[1], dtype=tf.float32)),
                       shape=[]),
            lambda: tf.one_hot(segment_label[i] - 1, 2885, on_value=predict_sbp[i][0],
                                                   off_value=0.0, dtype=tf.float32), lambda:
                tf.one_hot(segment_label[i] - 1, 2885, on_value=0.5*(predict_sbp[i][0]-segment_benchmack[segment_label[i] - 1]),
                                               off_value=0.0,
                                               dtype=tf.float32))
        segment_benchmack = segment_benchmack + condition

        if condition:
            one_hot = tf.one_hot(i, 2885, on_value=1.0, off_value=0.0, dtype=tf.float32)
            segment_benchmack = segment_benchmack + one_hot
        else:
            one_hot_2 = tf.one_hot(i, 2885, on_value=3.0, off_value=0.0, dtype=tf.float32)
            half_one_hot = tf.one_hot(i, 2885, on_value=0.5, off_value=0.0, dtype=tf.float32)
            segment_benchmack = segment_benchmack + one_hot_2
            # benchmack_copy = tf.multiply(benchmack_copy, half_one_hot)
    '''

    def loss_function(output, segment_benchmack):
        # benchmack = benchmack + tf.ones([2885], dtype=tf.float32)
        # predict_sbp = vgg_inference.inference_op(input, 0.5)
        mse_loss = tf.losses.mean_squared_error(output, Y)

        batch_benchmack = tf.gather(segment_benchmack, segment_label - 1)
        segment_loss = tf.losses.mean_squared_error(output, tf.reshape(batch_benchmack, shape=[-1, 1]))
        loss = mse_loss + 0.01*segment_loss

        tf.add_to_collection('losses', loss)
        losses = tf.add_n(tf.get_collection('losses'))
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_op = optimizer.minimize(losses)

        return train_op, mse_loss, segment_loss, loss

    train_X = load_data.get_trainset('train_set_v8_random_5.h5', 'ppw1')
    train_Y = load_data.get_trainset('train_set_v8_random_5.h5', 'label_nsbp_r')
    train_average_correct = load_data.get_trainset('train_set_v8_random_5.h5', 'subject_sbp')
    added_train_X = np.hstack((train_X, train_average_correct)).reshape([-1, 1, 501, 1])
    train_X_segment_idx = load_data.get_trainset('train_set_v8_random_5.h5', 'segment').reshape(-1)
    train_X_segment_idx = np.asarray(train_X_segment_idx, dtype=np.int32)
    train_Y = train_Y.reshape([-1, 1])
    # print(train_Y.shape)
    test_X = load_data.get_testset('test_set_v8_random_5.h5', 'ppw1')
    test_Y = load_data.get_testset('test_set_v8_random_5.h5', 'label_nsbp_r')
    test_Y_segment_idx = load_data.get_testset('test_set_v8_random_5.h5', 'segment').reshape(-1)
    test_Y_segment_idx = np.array(test_Y_segment_idx, dtype=np.int32)
    # feed_test_Y_segment_idx = test_Y_segment_idx.reshape([-1, 1])
    test_average_correct = load_data.get_testset('test_set_v8_random_5.h5', 'subject_sbp')
    idx_testdata = load_data.get_testset('test_set_v8_random_5.h5', 'testset_idx')
    added_test_X = np.hstack((test_X, test_average_correct)).reshape([-1, 1, 501, 1])
    test_Y = test_Y.reshape([-1, 1])

    # print(added_train_X)
    # exit(0)

    train_op, mse_loss, train_segment_loss, cost = loss_function(predict_sbp, segment_benchmack)
    _, test_mse_loss, test_segment_loss, test_cost = loss_function(predict_sbp, segment_benchmack)

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
            shuffle_train_X = added_train_X[idx_array]
            # print('shuffle_train:{}'.format())
            shuffle_train_Y = train_Y[idx_array]
            shuffle_train_X_segment_idx = train_X_segment_idx[idx_array]
            # shuffle_x_train = x_train[idx_array]
            # shuffle_y_train = y_train[idx_array]
            for i in range(step):
                start_time = time.time()
                start = i * batch_size
                end = start + batch_size
                # print(shuffle_train_X_segment_idx[start:end].shape)
                # print(shuffle_train_X_segment_idx[start:end].dtype)
                feed_dict = {X: shuffle_train_X[start:end], Y: shuffle_train_Y[start:end],
                             segment_label: shuffle_train_X_segment_idx[start:end]}
                # print('X:{}'.format(len(feed_dict[X])))
                _, loss, mse_cost, benchmack, segmentids, segment_cost, train_predict = sess.run(
                    [train_op, cost, mse_loss, segment_benchmack, segment_label, train_segment_loss, predict_sbp],
                    feed_dict=feed_dict)
                duration = time.time() - start_time
                print(
                    'Epoch {0} step {1}/{2}: train_loss:{3:.6f}  mse:{4:.6f}  segment_loss:{6:.6f} ({5:.3f} sec/step)'.format(
                        epoch, i + 1, step,
                        loss, mse_cost,
                        duration, segment_cost))

                # variable_name = [v.name for v in tf.trainable_variables()]
                # np.savetxt('test_2/benchmack-epoch_' + str(i+1) + '.txt', benchmack, fmt='%.2f')
                # np.savetxt('test_2/predict-epoch_' + str(i + 1) + '.txt', train_predict, fmt='%.2f')
                # np.savetxt('test_2/segments_' + str(i+1) + '.txt', segmentids, fmt='%d')
            # exit(2)
            if epoch % 50 == 0:
                np.savetxt('test_3/benchmack-epoch_' + str(epoch) + '.txt', benchmack, fmt='%.2f')
                # exit(2)
                predict, total_loss, mse = sess.run(
                    [predict_sbp, test_cost, test_mse_loss],
                    feed_dict={X: added_test_X, Y: test_Y,
                               segment_label: test_Y_segment_idx})
                print('mse_loss in testset:{0:.6f}\ntotal_loss:{1:.6f}\n'.format(mse, total_loss))
                # np.savetxt('test_3/benchmack-epoch_' + str(epoch) + '-after_test.txt', aftertest_benchmack, fmt='%.2f')
                dict = {}
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
                correct_mse = np.mean(np.square(predict - test_Y))
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
                plt.text(100, 100,
                         'lr={0},keepprob={1},batchsize={2},t_epoch={3}\nmse_loss:{4:.6f}\nRMSE:{5:.6f}\nStd:{6:.6f}\nME:{7:.6f}\nMAE:{8:.6f}\ntotal_loss:{9:.6f} \n'.format(
                             lr, kprob, batch_size, 1000, correct_mse, rmse, std, me, mae, total_loss))
                if os.path.exists(dir_path) == False:
                    os.makedirs(dir_path)
                np.savetxt(dir_path + '/predict_test_epoch-' + str(epoch) + '.txt', predict, fmt='%d')
                plt.savefig(dir_path + '/epoch-' + str(epoch))
                plt.close()
                # exit(2)
                if epoch == 1000:
                    with open(dir_path + '/result.txt', 'w', encoding='utf-8') as f:
                        f.writelines('index\tidx_data\ttest_label\tpredict\n')
                        for i in range(len(predict)):
                            f.writelines('{0}\t{1}\t{2}\t{3:.1f}\n'.format(i + 1, int(idx_testdata[i].tolist() + 1),
                                                                           test_Y[i][0].tolist(),
                                                                           predict[i][0].tolist()))
                        f.writelines('MSE:{0:.6f}\n'.format(correct_mse))
                        f.writelines('RMSE:{0:.6f}\n'.format(rmse))
                        f.writelines('Std:{0:.6f}\n'.format(std))
                        f.writelines('ME:{0:.6f}\n'.format(me))
                        f.writelines('MAE:{0:.6f}\n'.format(mae))
                    # exit(3)

    print('Train Finished !!!')


def main():
    lr = 1e-4
    keepprob = 0.8
    # segment_benchmack = tf.zeros(shape=[2885], dtype=tf.float32)
    training(1024, lr, keepprob)


if __name__ == '__main__':
    main()
    # a = get_segment_average('alldata_v2.mat')
    # print(a)
