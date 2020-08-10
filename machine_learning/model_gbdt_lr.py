#! /usr/bin/env python
# -*- coding:utf8 -*-
from __future__ import absolute_import, division, print_function

import random

import pandas as pd
import tensorflow as tf
from sklearn import metrics
# 加入模型训练
from sklearn.linear_model import LogisticRegression
# %matplotlib inline
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from common_tool import config_manager as cfg

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier, GradientBoostingClassifier)
from sklearn.externals.six import StringIO
from sklearn import tree
import pydotplus
from IPython.display import Image
from sklearn.tree import DecisionTreeClassifier,export_graphviz


import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# 导入后加入以下列，再显示时显示完全。
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

sample_path = cfg.get_config('file', 'sample_path')
print('sample_path:', sample_path)
print(os.path.abspath(__file__))
print(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)
print(sys.path)
# PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
# print(PROJECT_ROOT)
# file_path = os.path.join(PROJECT_ROOT,"data\\edge\\0_fuse.txt")
data_train = pd.read_csv('..\\feature_project\\sample_result\\2020-07-02_102303log_normal_train.csv')
data_test = pd.read_csv('..\\feature_project\\sample_result\\2020-07-02_102303log_normal_test.csv')
# 通过自定义的函数对数据进行转换
data_train_fill = data_train.copy()
data_test_fill = data_test.copy()

features_X_train = ['cur_same_prov', 'charge_wait_time', 'x_in_chargepolicy', 'x_in_commlog5', 'cur_vcode_time_max',
                   'cur_vcode_time_min', 'cur_vcode_time_mean', 'cur_vcode_time_sum', 'x_ext17', 'x_in_commlog29',
                   'cur_dot_ct', 'x_ext16', 'cur_xff_ct', 'cur_ipisp', 'x_ext6', 'web_ch_charge_count_day',
                   'web_ch_charge_elseWhereRatio_day', 'web_ch_charge_phoneFromNum_day', 'web_ch_charge_ipFromNum_day',
                   'web_ch_charge_ipTopRatio_day', 'web_ch_charge_clickT10Ratio_day', 'web_ch_charge_clickRatio_day',
                   'web_ch_charge_vsRatio_day', 'wseb_ch_charge_uaT5Ratio_day', 'web_ch_charge_uaRatio_day',
                   'web_ch_charge_ctime_ret1_ratio_day', 'web_ch_charge_ctime_ret3_ratio_day',
                   'web_ch_charge_ctime_ret5_ratio_day', 'web_ch_charge_ydRatio_day', 'web_ch_charge_ltRatio_day',
                   'web_ch_charge_dxRatio_day', 'web_ch_charge_wapRatio_day', 'web_ch_charge_phone_avgct_day',
                   'web_ch_5day_top60min_every_day_count', 'web_ch_5day_top60min_every_day_ratio',
                   'web_ch_charge_phone_provT1Ratio_day', 'web_ch_charge_foreignip_count_day',
                   'web_ch_charge_selfnet_ratio_day', 'web_ch_charge_across_city_ratio_day',
                   'web_ch_charge_fpay_ratio_day', 'web_ch_charge_conversion_ratio_day',
                   'web_ch_charge_ip_avg_charge_day', 'web_phone_distinct_ua_day', 'web_phone_success_count_day',
                   'web_phone_total_count_day', 'web_phone_chargetype0_count_day', 'web_phone_chargetype1_count_day',
                   'web_phone_succ_amount_day', 'web_phone_ip_diff_prov_ratio_day', 'app_phone_success_count_day',
                   'app_phone_total_count_day', 'app_phone_chargetype0_count_day', 'app_phone_chargetype1_count_day',
                   'app_phone_succ_amount_day', 'web_ip_total_count_day', 'web_ip_succ_distinct_ch_day',
                   'web_ip_phone_diff_prov_ratio_day', 'web_ip_distinct_hour_day', 'web_ip_distinct_ua_day',
                   'web_ip_ply_charge_ratio_day', 'web_ip_charge_succ_ratio_day', 'web_ip_distinct_succ_phone_day',
                   'app_ip_total_count_day', 'app_ip_succ_distinct_ch_day', 'app_ip_distinct_hour_day',
                   'app_ip_charge_succ_ratio_day', 'app_ip_distinct_succ_phone_day', 'web_ch_charge_count_week',
                   'web_ch_charge_elseWhereRatio_week', 'web_ch_charge_phoneFromNum_week',
                   'web_ch_charge_ipFromNum_week', 'web_ch_charge_suc_ratio_week', 'web_ch_charge_ip_avg_charge_week',
                   'web_phone_distinct_ua_week', 'web_phone_success_count_week', 'web_phone_total_count_week',
                   'web_phone_chargetype0_count_week', 'web_phone_chargetype1_count_week', 'web_phone_succ_amount_week',
                   'web_phone_ip_diff_prov_ratio_week', 'app_phone_success_count_week', 'app_phone_total_count_week',
                   'app_phone_chargetype0_count_week', 'app_phone_chargetype1_count_week', 'app_phone_succ_amount_week',
                   'web_ip_total_count_week', 'web_ip_succ_distinct_ch_week', 'web_ip_phone_diff_prov_ratio_week',
                   'web_ip_ply_charge_ratio_week', 'web_ip_charge_succ_ratio_week', 'web_ip_distinct_succ_phone_week',
                   'app_ip_total_count_week', 'app_ip_succ_distinct_ch_week', 'app_ip_charge_succ_ratio_week',
                   'app_ip_distinct_succ_phone_week', 'web_ch_charge_count_month', 'web_ch_charge_elseWhereRatio_month',
                   'web_ch_charge_phoneFromNum_month', 'web_ch_charge_ipFromNum_month',
                   'web_ch_charge_ipTopRatio_month', 'web_ch_charge_clickT10Ratio_month',
                   'web_ch_charge_clickRatio_month', 'web_ch_charge_vsRatio_month', 'web_ch_charge_uaT5Ratio_month',
                   'web_ch_charge_uaRatio_month', 'web_ch_charge_ydRatio_month', 'web_ch_charge_ltRatio_month',
                   'web_ch_charge_dxRatio_month', 'web_ch_charge_wapRatio_month', 'web_ch_charge_phone_avgct_month',
                   'web_ch_charge_phone_provT1Ratio_month', 'web_ch_charge_foreignip_count_month',
                   'web_ch_charge_selfnet_ratio_month', 'web_ch_charge_ip_avg_charge_month',
                   'web_phone_distinct_ua_month', 'web_phone_success_count_month', 'web_phone_total_count_month',
                   'web_phone_chargetype0_count_month', 'web_phone_chargetype1_count_month',
                   'web_phone_succ_amount_month', 'web_phone_ip_diff_prov_ratio_month', 'app_phone_total_count_month',
                   'app_phone_chargetype0_count_month', 'app_phone_chargetype1_count_month',
                   'app_phone_succ_amount_month', 'web_ip_total_count_month', 'web_ip_succ_distinct_ch_month',
                   'web_ip_phone_diff_prov_ratio_month', 'web_ip_ply_charge_ratio_month',
                   'web_ip_charge_succ_ratio_month', 'web_ip_distinct_succ_phone_month', 'app_ip_total_count_month',
                   'app_ip_succ_distinct_ch_month', 'app_ip_charge_succ_ratio_month',
                   'app_ip_distinct_succ_phone_month', 'web_ip_10min_suc_ct', 'web_ip_10min_channel_ct',
                   'web_ip_10min_phone_ct', 'web_ip_1h_ua_ct', 'web_ch_10min_ct', 'web_ch_30min_ct',
                   'web_ch_10min_suc_rt', 'web_ch_10min_fpay_ct', 'web_ch_10min_fpay_rt', 'web_ch_30min_phone_ew_ct',
                   'web_ch_30min_phone_provice_ct', 'web_ch_30min_ip_provice_ct', 'web_ch_30min_click_pt_top5_rt',
                   'web_ch_30min_click_pt_ct', 'web_ch_30min_ua_top5_rt', 'web_ch_30min_ua_norepeat_rt',
                   'web_phone_10min_charge_ct', 'web_phone_10min_change_address', 'app_phone_10min_charge_ct']

featrures_X_del = []


def model_dbdt_lr_fit(train_x, train_y, features_list):
    n_estimator = 10
    # 带入模型进行训练
    dbdt_model = GradientBoostingClassifier(n_estimators=n_estimator,min_samples_split=2000,
                                            min_samples_leaf=500,max_depth=5,
                                            max_features='sqrt',subsample=0.8,learning_rate=0.1)
    dbdt_model.fit(train_x, train_y)
    train_x_new = dbdt_model.apply(train_x)[:, :, 0]
    print('after gbdt train_x_new shape', train_x_new.shape)
    print('after gbdt train_x_new data\n', train_x_new[0:3, :])

    one_hot_enc = OneHotEncoder()
    one_hot_enc.fit(train_x_new)
    data_x_new_onehot = one_hot_enc.transform(train_x_new)
    print('after onehot data_x_new_onehot shape', data_x_new_onehot.shape)


    lr_model = LogisticRegression()
    lr_model.fit(data_x_new_onehot, train_y)
    print('模型实际迭代次数：', lr_model.n_iter_)
    features_weight = lr_model.coef_[0]
    df_weight = pd.DataFrame()
    for i in range(0, data_x_new_onehot.shape[1]):
        # print('特征：%s，系数:%f' % (features_list[i], features_weight[i]))
        new = pd.DataFrame(data={'feature': str(i) , 'weight': [features_weight[i]]})
        df_weight = df_weight.append(new, ignore_index=True)
    # print('系数：', lr_model.coef_)
    print('截距：', lr_model.intercept_)
    lr_model_score = lr_model.score(data_x_new_onehot, train_y)
    print('模型得分：', lr_model_score)
    return lr_model, df_weight, dbdt_model, one_hot_enc


def model_lr_metric_info(model_lr, model_dbdt, one_hot_enc, data_x, data_y, data_desc=None):
    data_x_new = model_dbdt.apply(data_x)[:, :, 0]
    data_x_new_onehot = one_hot_enc.transform(data_x_new)

    pred_y = model_lr.predict_proba(data_x_new_onehot)[:, 1]
    pred_class_y = model_lr.predict(data_x_new_onehot)
    conf_matrix = confusion_matrix(data_y, pred_class_y)
    accuracy = accuracy_score(data_y, pred_class_y)  # 准确率
    precision = precision_score(data_y, pred_class_y)  # 精准率
    recall = recall_score(data_y, pred_class_y)  # 召回率
    f1score = f1_score(data_y, pred_class_y)
    print('%s混淆矩阵：\n%s' % (data_desc, conf_matrix))
    print('%s准确率：%f' % (data_desc, accuracy))
    print('%s精准率：%f' % (data_desc, precision))
    print('%s召回率：%f' % (data_desc, recall))
    print('%sf1score：%f' % (data_desc, f1score))
    fpr_lr, tpr_lr, threshold = roc_curve(data_y, pred_y)
    rocauc = metrics.auc(fpr_lr, tpr_lr)  # 计算AUC
    print('%sAUC：%f' % (data_desc, rocauc))
    calc_ks(tpr_lr, fpr_lr, threshold, data_desc)
    return fpr_lr, tpr_lr, rocauc, threshold



def view_roc(fpr_lr, tpr_lr, rocauc, desc=None):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置
    plt.plot(fpr_lr, tpr_lr, label='%s LR (AUC=%0.2f)' % (desc, rocauc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim = ([0.0, 1.0])
    plt.ylim = ([0.0, 1.0])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('%s ROC Curve' % desc)
    plt.legend(loc='best')
    plt.show()


def view_ks(fpr_lr, tpr_lr, threshold, desc=None):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置
    # KS值范围在0%-100%，判别标准如下：
    #
    # KS: <20% : 差
    # KS: 20%-40% : 一般
    # KS: 41%-50% : 好
    # KS: 51%-75% : 非常好
    # KS: >75% : 过高，需要谨慎的验证模型
    max_ks = abs(fpr_lr - tpr_lr).max()
    fig, ax = plt.subplots()
    ax.plot(1 - threshold, tpr_lr, label='tpr(True positive rate)')
    ax.plot(1 - threshold, fpr_lr, label='fpr(False positive rate)')
    ax.plot(1 - threshold, tpr_lr - fpr_lr, label='KS(max:%f)' % max_ks)
    plt.xlabel('score')
    plt.title('%s KS curve' % desc)
    plt.xlim = ([0.0, 1.0])
    plt.ylim = ([0.0, 1.0])
    plt.figure(figsize=(20, 20))
    ax.legend(loc='upper left')
    plt.show()


def prepare_data(features_use):
    # features_use = random.sample(features_use, 10)  # 从list中随机获取N个元素，作为一个片断返回
    # print("\n随机选取%d个特征:%s" % (len(features_use), str(features_use)))
    shape_len = len(features_use)
    print("\n最终用于模型训练的特征数量:", shape_len)

    fetures_Y_train = ['y_label']
    data_X_train = data_train_fill[features_use]
    data_Y_train = data_train_fill[fetures_Y_train]
    data_Y_train['y_label'] = data_Y_train['y_label'].astype(int)

    data_X_test = data_test_fill[features_use]
    data_Y_test = data_test_fill[fetures_Y_train]
    data_Y_test['y_label'] = data_Y_test['y_label'].astype(int)

    return data_X_train, data_Y_train, data_X_test, data_Y_test

def view_tree(clf, features_x, data_y):
    # dot_data = StringIO()
    # export_graphviz(clf, out_file=dot_data, feature_names=features_x,
    #                 class_names=data_y, filled=True,
    #                 rounded=True, special_characters=True)
    # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # graph.write_png('Telephone.png')
    # dot_data = tree.export_graphviz(clf, out_file=None,
    #                                 feature_names=features_x,
    #                                 filled=True,
    #                                 impurity=False, rounded=True)
    # import pydotplus
    # graph = pydotplus.graph_from_dot_data(dot_data)
    # graph.get_nodes()[7].set_fillcolor('#FFF2DD')
    # from IPython.display import Image
    # Image(graph.create_png())
    # graph.write_png('model_white_background.png')
    # https://blog.csdn.net/chai_zheng/article/details/78226556
    dot_data = StringIO()
    tree.export_graphviz(clf)
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=features_x,
                         class_names=data_y,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('aaa.png')

def model_execute(features_use):
    data_X_train, data_Y_train, data_X_test, data_Y_test = prepare_data(features_use)
    print('训练集特征shape：', data_X_train.shape)
    print('测试集特性shape：', data_X_test.shape)
    model_lr, df_weight, model_dbdt, one_hot_enc = model_dbdt_lr_fit(data_X_train, data_Y_train, features_use)
    print('model_dbdt模型信息：', model_dbdt)
    print('model_lr模型信息：', model_lr)
    print('one_hot_enc模型信息：', one_hot_enc)

    # we can access individual trees
    trees = model_dbdt.estimators_.ravel()
    print('tree num : ', len(trees))
    # view_tree(trees[0], features_use, 'y_label')


    fpr_lr, tpr_lr, rocauc, threshold = \
        model_lr_metric_info(model_lr, model_dbdt, one_hot_enc, data_X_train, data_Y_train, '训练集数据')
    view_roc(fpr_lr, tpr_lr, rocauc, '训练集数据')
    view_ks(fpr_lr, tpr_lr, threshold, '训练集数据')
    # ===============测试集=================

    fpr_lr_test, tpr_lr_test, rocauc_test, threshold_test = \
        model_lr_metric_info(model_lr, model_dbdt, one_hot_enc, data_X_test, data_Y_test, '测试集数据')
    view_roc(fpr_lr_test, tpr_lr_test, rocauc_test, '测试集数据')
    view_ks(fpr_lr_test, tpr_lr_test, threshold_test, '测试集数据')
    return df_weight


def calc_ks(tpr, fpr, thresholds, desc=None):
    # 计算ks
    KS_max = 0
    best_thr = 0
    for i in range(len(fpr)):
        if i == 0:
            KS_max = tpr[i] - fpr[i]
            best_thr = thresholds[i]
        elif tpr[i] - fpr[i] > KS_max:
            KS_max = tpr[i] - fpr[i]
            best_thr = thresholds[i]

    print('%s最大KS为：%f' %(desc, KS_max))
    print('%s最佳阈值为：%f' %(desc, best_thr))




def model_stay_features():
    features_use = ['charge_wait_time','web_ch_30min_click_pt_top5_rt','web_ch_30min_ua_norepeat_rt','web_ch_5day_top60min_every_day_ratio','web_ch_charge_ctime_ret1_ratio_day','web_ch_charge_ip_avg_charge_day','web_ch_charge_phone_provT1Ratio_day','web_ch_charge_uaRatio_month','web_ip_10min_suc_ct','web_ip_charge_succ_ratio_week','web_ip_distinct_succ_phone_day','web_ip_succ_distinct_ch_week','web_ip_total_count_week','web_phone_10min_charge_ct','web_phone_distinct_ua_day','web_phone_success_count_week','web_phone_total_count_week']
    # features_use = features_X_train
    print("\n需要的特征数量:", len(features_use))
    df = model_execute(features_use)
    df = df.reindex(df['weight'].abs().sort_values(ascending=False).index)
    print('=========TOP10全部特征========')
    print(df.head(10))


if __name__ == '__main__':
    # model_flow(
    model_stay_features()
