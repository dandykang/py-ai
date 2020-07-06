from __future__ import absolute_import, division, print_function

import matplotlib
import numpy as np
import random
import requests
import string
import tarfile
import tensorflow as tf
import pandas as pd
# %matplotlib inline
import matplotlib.pyplot as plt

import seaborn as sns
import copy
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import warnings
import os
from sunshine import models as mds
from sunshine import metricview as mview

print(tf.__version__)

tf.print("tensorflow version:", tf.__version__)

a = tf.constant("hello")
b = tf.constant("tensorflow2")
c = tf.strings.join([a, b], " ")
tf.print(c)
# PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
# print(PROJECT_ROOT)
# file_path = os.path.join(PROJECT_ROOT,"data\\edge\\0_fuse.txt")
data_train = pd.read_csv('..\\feature_project\\sample_result\\2020-07-02_102303log_normal_train.csv')
data_test = pd.read_csv('..\\feature_project\\sample_result\\2020-07-02_102303log_normal_test.csv')
# 通过自定义的函数对数据进行转换
data_train_fill = data_train.copy()
data_test_fill = data_test.copy()

fetures_X_train = ['web_ch_charge_phoneFromNum_day','cur_vcode_time_sum','web_ip_distinct_succ_phone_day','web_ch_charge_phone_provT1Ratio_day','web_ch_charge_ltRatio_day','web_ip_distinct_hour_day','web_ch_charge_conversion_ratio_day','web_ch_10min_fpay_ct','web_phone_succ_amount_week','web_ch_charge_uaRatio_day','x_ext6','web_ch_charge_across_city_ratio_day','charge_wait_time','web_ip_distinct_succ_phone_month','web_phone_chargetype1_count_week','web_ch_charge_phoneFromNum_month','cur_vcode_time_max','x_ext17','cur_vcode_time_mean','web_ch_30min_ua_norepeat_rt','web_ip_total_count_day','cur_vcode_time_min','x_in_commlog5','web_ip_ply_charge_ratio_month','web_ip_succ_distinct_ch_day','web_ip_distinct_succ_phone_week','web_ip_10min_phone_ct','web_ch_charge_phone_provT1Ratio_month','web_ip_10min_suc_ct','web_ch_charge_clickT10Ratio_month','web_ch_charge_foreignip_count_day','web_phone_total_count_day','web_phone_10min_change_address','web_ch_charge_ip_avg_charge_week','web_ch_charge_ltRatio_month','web_ch_charge_ip_avg_charge_day','x_in_chargepolicy','web_ip_total_count_month','web_ch_charge_uaRatio_month','web_ch_charge_clickT10Ratio_day','web_ch_charge_clickRatio_month','web_phone_total_count_week','web_phone_chargetype1_count_month','web_phone_success_count_week','web_ch_charge_phoneFromNum_week','web_phone_success_count_month','web_ch_charge_foreignip_count_month','web_ip_succ_distinct_ch_week','web_phone_total_count_month','web_ip_total_count_week','web_phone_10min_charge_ct']
# fetrures_X_del = ['cur_vcode_time_max','cur_vcode_time_min','cur_vcode_time_mean','cur_vcode_time_sum']
fetrures_X_del = ['charge_wait_time','charge_wait_time','cur_vcode_time_max','cur_vcode_time_mean','cur_vcode_time_sum','web_ch_charge_across_city_ratio_day','web_ch_charge_conversion_ratio_day','web_ch_charge_uaRatio_month','web_ip_ply_charge_ratio_month','web_ch_10min_fpay_ct','web_ch_charge_foreignip_count_day','web_ch_charge_foreignip_count_month','web_phone_10min_change_address','x_ext17','cur_vcode_time_max','cur_vcode_time_mean','cur_vcode_time_sum','web_ch_charge_conversion_ratio_day']
fetrures_X_06 = ['web_ch_5day_top60min_every_day_count','web_phone_ip_diff_prov_ratio_day','web_phone_ip_diff_prov_ratio_week','web_ch_charge_wapRatio_month','web_ch_charge_elseWhereRatio_day','web_ip_phone_diff_prov_ratio_day','web_ch_charge_elseWhereRatio_week','web_ip_phone_diff_prov_ratio_week','web_ch_charge_elseWhereRatio_month','web_phone_ip_diff_prov_ratio_month','web_ip_phone_diff_prov_ratio_month','x_in_commlog29','cur_dot_ct','x_in_commlog5','web_ch_charge_count_month','web_ch_charge_vsRatio_month','web_ch_30min_phone_ew_ct']
fetrures_X_07 = ['web_ch_charge_dxRatio_day','web_ch_charge_ydRatio_month','web_ch_charge_dxRatio_month','web_ch_30min_ua_top5_rt','web_ch_charge_count_day','web_ch_charge_ydRatio_day','web_ch_charge_count_week','web_ch_10min_ct','web_ch_30min_ct']
fetrures_X_08 = ['x_ext16']
fetrures_X_05 = ['x_in_commlog29','cur_dot_ct','x_ext16','cur_ipisp','web_ch_charge_count_day','web_ch_charge_elseWhereRatio_day','web_ch_charge_ipTopRatio_day','web_ch_charge_ydRatio_day','web_ch_charge_dxRatio_day','web_ch_charge_phone_avgct_day','web_ch_5day_top60min_every_day_count','web_ch_charge_selfnet_ratio_day','web_ch_charge_across_city_ratio_day','web_phone_chargetype0_count_day','web_phone_ip_diff_prov_ratio_day','web_ip_phone_diff_prov_ratio_day','web_ch_charge_count_week','web_ch_charge_elseWhereRatio_week','web_ch_charge_suc_ratio_week','web_phone_chargetype0_count_week','web_phone_ip_diff_prov_ratio_week','web_ip_phone_diff_prov_ratio_week','web_ch_charge_count_month','web_ch_charge_elseWhereRatio_month','web_ch_charge_vsRatio_month','web_ch_charge_ydRatio_month','web_ch_charge_dxRatio_month','web_ch_charge_wapRatio_month','web_ch_charge_selfnet_ratio_month','web_phone_distinct_ua_month','web_phone_chargetype0_count_month','web_phone_ip_diff_prov_ratio_month','web_ip_phone_diff_prov_ratio_month','web_ip_10min_phone_ct','web_ch_10min_ct','web_ch_30min_ct','web_ch_10min_suc_rt','web_ch_30min_phone_ew_ct','web_ch_30min_ua_top5_rt','x_ext16','cur_ipisp','web_ch_charge_count_day','web_ch_charge_elseWhereRatio_day','web_ch_charge_ipTopRatio_day','web_ch_charge_ydRatio_day','web_ch_charge_dxRatio_day','web_ch_charge_selfnet_ratio_day','web_phone_ip_diff_prov_ratio_day','web_ip_phone_diff_prov_ratio_day','web_ch_charge_count_week','web_ch_charge_elseWhereRatio_week','web_ch_charge_suc_ratio_week','web_phone_ip_diff_prov_ratio_week','web_ip_phone_diff_prov_ratio_week','web_ch_charge_elseWhereRatio_month','web_ch_charge_ipFromNum_month','web_ch_charge_vsRatio_month','web_ch_charge_ydRatio_month','web_ch_charge_dxRatio_month','web_ch_charge_selfnet_ratio_month','web_phone_ip_diff_prov_ratio_month','web_ip_phone_diff_prov_ratio_month','web_ch_30min_ua_top5_rt','x_in_commlog29','cur_dot_ct','x_ext16','cur_ipisp','web_ch_charge_count_day','web_ch_charge_elseWhereRatio_day','web_ch_charge_ydRatio_day','web_ch_charge_dxRatio_day','web_ch_5day_top60min_every_day_count','web_ch_charge_across_city_ratio_day','web_phone_chargetype0_count_day','web_phone_ip_diff_prov_ratio_day','web_ip_phone_diff_prov_ratio_day','web_ch_charge_count_week','web_ch_charge_elseWhereRatio_week','web_phone_chargetype0_count_week','web_phone_ip_diff_prov_ratio_week','web_ip_phone_diff_prov_ratio_week','web_ch_charge_count_month','web_ch_charge_elseWhereRatio_month','web_ch_charge_vsRatio_month','web_ch_charge_ydRatio_month','web_ch_charge_dxRatio_month','web_ch_charge_wapRatio_month','web_phone_chargetype0_count_month','web_phone_ip_diff_prov_ratio_month','web_ch_10min_ct','web_ch_30min_ct','web_ch_30min_phone_ew_ct','web_ch_30min_ua_top5_rt']
fetrures_X_05 = list(set(fetrures_X_05))
fetures_X_0102 = ['charge_wait_time','x_in_chargepolicy','x_in_commlog5','cur_vcode_time_max','cur_vcode_time_min','cur_vcode_time_mean','cur_vcode_time_sum','x_ext17','x_ext6','web_ch_charge_phoneFromNum_day','web_ch_charge_clickT10Ratio_day','web_ch_charge_uaRatio_day','web_ch_charge_ltRatio_day','web_ch_charge_phone_provT1Ratio_day','web_ch_charge_foreignip_count_day','web_ch_charge_ip_avg_charge_day','web_phone_success_count_day','web_phone_total_count_day','app_phone_success_count_day','app_phone_total_count_day','app_phone_chargetype1_count_day','app_phone_succ_amount_day','web_ip_total_count_day','web_ip_succ_distinct_ch_day','web_ip_distinct_hour_day','web_ip_distinct_succ_phone_day','app_ip_total_count_day','app_ip_succ_distinct_ch_day','app_ip_distinct_hour_day','app_ip_charge_succ_ratio_day','app_ip_distinct_succ_phone_day','web_ch_charge_phoneFromNum_week','web_ch_charge_ip_avg_charge_week','web_phone_success_count_week','web_phone_total_count_week','app_phone_success_count_week','app_phone_total_count_week','app_phone_chargetype0_count_week','app_phone_chargetype1_count_week','app_phone_succ_amount_week','web_ip_total_count_week','web_ip_succ_distinct_ch_week','web_ip_distinct_succ_phone_week','app_ip_total_count_week','app_ip_succ_distinct_ch_week','app_ip_charge_succ_ratio_week','app_ip_distinct_succ_phone_week','web_ch_charge_phoneFromNum_month','web_ch_charge_clickT10Ratio_month','web_ch_charge_uaRatio_month','web_ch_charge_phone_provT1Ratio_month','web_ch_charge_foreignip_count_month','web_phone_success_count_month','app_phone_total_count_month','app_phone_chargetype0_count_month','app_phone_chargetype1_count_month','app_phone_succ_amount_month','web_ip_total_count_month','web_ip_10min_suc_ct','web_ch_30min_ua_norepeat_rt','web_phone_10min_charge_ct','web_phone_10min_change_address']
fetures_X_00to01 = ['web_ch_charge_uaRatio_month','web_ch_charge_phone_provT1Ratio_month','web_ch_charge_foreignip_count_month','app_phone_chargetype1_count_month','web_ip_10min_suc_ct','web_ch_30min_ua_norepeat_rt','app_phone_chargetype1_count_day','app_phone_succ_amount_day','web_ip_distinct_hour_day','app_ip_total_count_day','app_ip_succ_distinct_ch_day','app_ip_charge_succ_ratio_day','app_ip_distinct_succ_phone_day','web_ch_charge_phoneFromNum_week','web_phone_success_count_week','app_phone_success_count_week','app_phone_chargetype0_count_week','app_phone_chargetype1_count_week','app_phone_succ_amount_week','web_ch_charge_phoneFromNum_month','web_ch_charge_clickT10Ratio_month','charge_wait_time','x_in_commlog5','cur_vcode_time_max','cur_vcode_time_mean','cur_vcode_time_sum','web_ch_charge_phoneFromNum_day','web_ch_charge_clickT10Ratio_day','web_ch_charge_uaRatio_day','web_ch_charge_phone_provT1Ratio_day','web_ch_charge_foreignip_count_day','web_phone_total_count_day','app_phone_success_count_day','app_phone_total_count_day','web_ch_charge_uaRatio_month','web_ch_charge_ltRatio_month','web_ch_charge_phone_provT1Ratio_month','web_ch_charge_foreignip_count_month','web_phone_success_count_month','web_phone_total_count_month','app_phone_total_count_month','app_phone_chargetype0_count_month','app_phone_chargetype1_count_month','app_phone_succ_amount_month','app_ip_total_count_month','app_ip_succ_distinct_ch_month','app_ip_distinct_succ_phone_month','web_ip_10min_suc_ct','web_phone_10min_charge_ct','web_phone_10min_change_address','app_phone_chargetype1_count_day','app_phone_succ_amount_day','app_ip_total_count_day','app_ip_succ_distinct_ch_day','app_ip_distinct_succ_phone_day','web_phone_success_count_week','web_phone_total_count_week','app_phone_success_count_week','app_phone_total_count_week','app_phone_chargetype0_count_week','app_phone_chargetype1_count_week','app_phone_succ_amount_week','app_ip_total_count_week','app_ip_succ_distinct_ch_week','app_ip_distinct_succ_phone_week','web_ch_charge_clickRatio_month','charge_wait_time','x_in_commlog5','cur_vcode_time_max','cur_vcode_time_mean','cur_vcode_time_sum','x_ext17','x_ext6','web_ch_charge_phoneFromNum_day','web_ch_charge_ltRatio_day','web_ch_charge_phone_provT1Ratio_day','web_ch_charge_foreignip_count_day','web_phone_total_count_day','app_phone_success_count_day','app_phone_total_count_day','web_ch_charge_uaRatio_month','web_ch_charge_phone_provT1Ratio_month','web_ch_charge_foreignip_count_month','app_phone_chargetype1_count_month','web_ip_10min_suc_ct','web_ch_30min_ua_norepeat_rt','app_phone_chargetype1_count_day','app_phone_succ_amount_day','web_ip_distinct_hour_day','app_ip_total_count_day','app_ip_succ_distinct_ch_day','app_ip_charge_succ_ratio_day','app_ip_distinct_succ_phone_day','web_ch_charge_phoneFromNum_week','web_phone_success_count_week','app_phone_success_count_week','app_phone_chargetype0_count_week','app_phone_chargetype1_count_week','app_phone_succ_amount_week','web_ch_charge_phoneFromNum_month','web_ch_charge_clickT10Ratio_month','charge_wait_time','x_in_commlog5','cur_vcode_time_max','cur_vcode_time_mean','cur_vcode_time_sum','web_ch_charge_phoneFromNum_day','web_ch_charge_clickT10Ratio_day','web_ch_charge_uaRatio_day','web_ch_charge_phone_provT1Ratio_day','web_ch_charge_foreignip_count_day','web_phone_total_count_day','app_phone_success_count_day','app_phone_total_count_day']

# print("\n原始样本特征数量:", len(fetures_X_train))
features_use = list(set(fetures_X_train) - set(fetrures_X_del))
# features_use = list(set(fetures_X_train) )
print("\n删除不需要的特征后数量:", len(features_use))

# features_use = list(set(features_use) - set(fetrures_X_05))
# print("\n删除大于0.5的特征后数量:", len(features_use))
# features_use = list(set(features_use) - set(fetrures_X_06))
# print("\n删除大于0.6的特征后数量:", len(features_use))
# features_use = list(set(features_use) - set(fetrures_X_07))
# print("\n删除大于0.7的特征后数量:", len(features_use))
# features_use = list(set(features_use) - set(fetrures_X_08))
# print("\n删除大于0.8的特征后数量:", len(features_use))

# features_use = ['charge_wait_time','web_ch_30min_ua_norepeat_rt','web_ch_charge_clickRatio_month',
#                 'web_ch_charge_ltRatio_day','web_ch_charge_phone_provT1Ratio_day','web_ch_charge_phoneFromNum_day',
#                 'web_ch_charge_uaRatio_month','web_ip_10min_suc_ct','web_ip_distinct_hour_day',
#                 'web_phone_10min_charge_ct']
# web_ch_charge_clickT10Ratio_month\web_ch_charge_clickT10Ratio_day\web_ch_charge_ltRatio_month\
# web_ch_charge_phone_provT1Ratio_month\web_ch_charge_phoneFromNum_month\web_ch_charge_phoneFromNum_week\
# web_ch_charge_uaRatio_day\'x_ext6'
features_use = random.sample(features_use, 10) #从list中随机获取N个元素，作为一个片断返回
print("\n随机选取%d个特征:%s" %(len(features_use),str(features_use)))

fetures_Y_train = ['y_label']
data_X_train = data_train_fill[features_use]
data_Y_train = data_train_fill[fetures_Y_train]
data_Y_train['y_label'] = data_Y_train['y_label'].astype(int)


data_X_test = data_test_fill[features_use]
data_Y_test = data_test_fill[fetures_Y_train]
data_Y_test['y_label'] = data_Y_test['y_label'].astype(int)

shape_len = len(features_use)
print("\n最终用于模型训练的特征数量:", shape_len)


# 定义模型
def create_model():
    input_tensor = tf.keras.Input(shape=(151,), name="input")
    x = tf.keras.layers.Dense(64, activation='relu')(input_tensor)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid', name="output_sigmoid")(x)
    model = tf.keras.Model(inputs=input_tensor, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=[tf.keras.metrics.AUC(name='auc'),
                           tf.keras.metrics.Recall(name='recall'),
                           tf.keras.metrics.Precision(name='precision')
                           ])
    return model


# 定义输入的shape
input_shape = (shape_len,)
epochs_times = 10
# # 每一层的神经元个数，此处时有三层
dense_info = [(5, 'relu'), (5, 'relu'), (1, tf.keras.activations.sigmoid)]
# # 使用封装方法获取型
model = mds.ModelSequential(input_shape, dense_info).build()
# 建立序列模型
# model=tf.keras.Sequential()
# # 添加隐藏层,神经元为6个，输入类型为一维数组的5个特征
# model.add(tf.keras.layers.Dense(64, input_shape=(154,), activation=tf.keras.activations.relu))
# model.add(tf.keras.layers.Dense(64, activation='relu'))
# model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid))
# model = create_model()
model.summary()

# 优化器【optimizer】：
# Adam 算法，即一种对随机目标函数执行一阶梯度优化的算法，该算法基于适应性低阶矩估计。
# 有很高的计算效率和较低的内存需求
# 损失函数【loss】：
# 常用的损失函数有：MSE均方误差、binary_crossentropy交叉熵损失函数、categorical_crossentropy分类交叉熵函数
# link： https://blog.csdn.net/legalhighhigh/article/details/81409551
#
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy',tf.keras.metrics.AUC(name='auc'),
                           tf.keras.metrics.Recall(name='recall'),
                           tf.keras.metrics.Precision(name='precision')])


# 添加callback
logdir = os.path.join("callback-1")
if not os.path.exists(logdir):
    os.mkdir(logdir)
model_file = os.path.join(logdir, "model_test_1.h5")

callbacks = [
    tf.keras.callbacks.TensorBoard(logdir),
    tf.keras.callbacks.ModelCheckpoint(model_file, save_best_only=True)
]
# 查看方式：进到虚拟环境，进行启动。(TF_KDL) E:\Working Code\TensorFlow\神经网络-练习>tensorboard --logdir=callback


# 带入训练，次数设置100次
# history = model.fit(data_X_train, data_Y_train, epochs=50)
# validation_split 分割一部分训练数据用于验证(也可手动划分集合进行设置validation_data=(data_X_test,data_Y_test),)
history = model.fit(data_X_train, data_Y_train, epochs=epochs_times,
                    validation_split=0.2,
                    callbacks=callbacks)

for key in history.history:
    print(key)
# 查看训练集、测试集 损失、正确率的对比
mview.plot_metric(history, "loss")
mview.plot_metric(history, "accuracy")
mview.plot_metric(history, "auc")
mview.plot_metric(history, "recall")
mview.plot_metric(history, "precision")

# 对测试集整体预测结果
eloss,eaccuracy,eauc,erecall,eprecision = model.evaluate(data_X_test, data_Y_test)
print("\n测试数据验证结果:")
print('\t损失值:', eloss)
print('\t正确率:', eaccuracy)
print('\tAUC值:', eauc)
print('\t召回率:', erecall)
print('\t精准度:', eprecision)
# 预测
predict_test = model.predict(data_X_test)
print('整体预测结果:%s' % predict_test)

# 预测类别
predict_test_classes = model.predict_classes(data_X_test)
print('整体预测类别结果:%s' % predict_test_classes)

# 预测单个需要对数据进行转换，转成1维5列的数据
# [[2.2 3.3 3.  9.1 4.5]]
# predict_one_X = np.array([2.00,3.00,1000.00,8.00,3.0])
# predict_one_X = predict_one_X.reshape(1,5)
# predict_one_Y = model.predict(predict_one_X)
# print('预测单个结果:%s' %predict_one_Y)
#
# predict_one_X_1 = np.array([[2.00,3.00,1000.00,8.00,3.0]])
# predict_one_Y_1 = model.predict(predict_one_X_1)
# print('预测单个结果:%s' %predict_one_Y_1)
