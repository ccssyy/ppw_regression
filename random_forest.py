import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import load_data

import matplotlib.pyplot as plt

import os

if __name__ == '__main__':
    # trainset = load_data.get_trainset('train_set.h5')
    # testset = load_data.get_testset('test_set.h5')
    train_X = load_data.get_trainset('train_set.h5', 'ppw1')
    train_Y = load_data.get_trainset('train_set.h5', 'label_nsbp_r')
    test_X = load_data.get_testset('test_set.h5', 'ppw1')
    test_Y = load_data.get_testset('test_set.h5', 'label_nsbp_r')

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

    # 随机森林回归模型
    rfr = RandomForestRegressor()

    rfr.fit(x_train, y_train)
    rfr_y_predict = rfr.predict(x_test)

    print('随机森林默认评估值：{0}'.format(rfr.score(x_test, y_test)))
    print('R_squared值：{0}'.format(r2_score(y_test, rfr_y_predict)))
    print('MSE:{0}'.format(mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rfr_y_predict))))
    print('y_test:{}'.format(ss_y.inverse_transform(y_test)))
    print('predict:{}'.format(ss_y.inverse_transform(rfr_y_predict)))
    plt.figure(figsize=(5, 500))
    plt.plot(np.arange(len(y_test)), ss_y.inverse_transform(y_test), 'b-', np.arange(len(rfr_y_predict)), ss_y.inverse_transform(rfr_y_predict), 'r:')
    plt.show()
