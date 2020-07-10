#! /usr/bin/env python
# -*- coding:utf8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from scipy import stats
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
print(os.path.abspath(__file__))
print(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)
print(sys.path)
from common_tool import config_manager as cfg
from common_tool import logger
import time

log_path = cfg.get_config('file', 'log_path')
log = logger.Logger(log_path + 'plot_tool.log', level='debug', when='H')


def ratio_stats(input_df, col):
    # 创建子图及间隔设置
    f, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(col, data=input_df)
    plt.show()

    num_1 = input_df[col].sum()
    num_0 = input_df[col].count() - input_df[col].sum()
    return num_0, num_1


def column_describe(input_df, col):
    col_desc = str(input_df[col].describe())
    f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 5))
    sns.distplot(input_df[col], kde_kws={"label": col_desc}, ax=ax1)
    sns.boxplot(y=col, data=input_df, ax=ax2)

    plt.savefig('sample_pic\\' + col + '.png')
    plt.show()


def column_compare_subplot(ori_df, normal_df, compare_col_list, y_label, pic_name):
    positive_nor_df = []
    negative_nor_df = []
    df_col_list = ori_df.columns.values.tolist()
    positive_ori_df = ori_df[ori_df[y_label] == 1]
    negative_ori_df = ori_df[ori_df[y_label] == 0]
    if (len(normal_df) > 0):
        positive_nor_df = normal_df[normal_df[y_label] == 1]
        negative_nor_df = normal_df[normal_df[y_label] == 0]

    log.logger.info('分析数据中，label_0 数量为：%i,label_1 数量为：%i,label_0 占比例为：%.2f%%'
                    % (len(negative_ori_df), len(positive_ori_df),
                       (len(negative_ori_df) / (len(negative_ori_df) + len(positive_ori_df)) * 100)))
    for compare_col in compare_col_list:
        if compare_col in df_col_list:
            column_compare_subplot2(positive_ori_df, negative_ori_df, positive_nor_df, negative_nor_df,
                                    compare_col, pic_name)
        else:
            log.logger.info('分析数据中，不包含column：%s 的列' % (compare_col))


def column_compare_subplot2(positive_ori_df, negative_ori_df, positive_nor_df, negative_nor_df, col, pic_name):
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置
        col_desc1 = str(positive_ori_df[col].describe(percentiles=[.1, .25, .5, .75, .90, .95]))
        col_desc2 = str(negative_ori_df[col].describe(percentiles=[.1, .25, .5, .75, .90, .95]))
        col_desc3 = str(positive_nor_df[col].describe(percentiles=[.1, .25, .5, .75, .90, .95]))
        col_desc4 = str(negative_nor_df[col].describe(percentiles=[.1, .25, .5, .75, .90, .95]))
        if len(positive_nor_df) > 0 and len(negative_nor_df) > 0:
            f, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(12, 10))
            # 设置子标题
            ax1.set_title('原始数据-正')
            ax2.set_title('原始数据-负')
            ax3.set_title('最终数据-正')
            ax4.set_title('最终数据-负')
            sns.distplot(positive_ori_df[col], kde_kws={"label": col_desc1}, ax=ax1)
            sns.distplot(negative_ori_df[col], kde_kws={"label": col_desc2}, ax=ax2)
            sns.distplot(positive_nor_df[col], kde_kws={"label": col_desc3}, ax=ax3)
            sns.distplot(negative_nor_df[col], kde_kws={"label": col_desc4}, ax=ax4)

            plt.savefig('sample_pic\\compare\\' + pic_name + '_compare_' + col + '.png')
            # plt.show()
        else:
            f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 5))
            # 设置子标题
            ax1.set_title('原始数据-正')
            ax2.set_title('原始数据-负')
            sns.distplot(positive_ori_df[col], kde_kws={"label": col_desc1}, ax=ax1)
            sns.distplot(negative_ori_df[col], kde_kws={"label": col_desc2}, ax=ax2)

            plt.savefig('sample_pic\\compare\\' + pic_name + '_compare_' + col + '.png')
            # plt.show()

    except Exception as e:
        log.logger.error('str(Exception):\t', str(Exception))
        log.logger.error('str(e):\t\t', str(e))
        log.logger.error('出现问题的列为%s:\t\t' % str(col))


def hist_count_show(df1, df2, col):
    kwargs = dict(histtype='stepfilled', alpha=0.3, bins=5, density=True)
    plt.hist(df1[col], **kwargs)
    plt.hist(df2[col], **kwargs)
    plt.show()


if __name__ == '__main__':
    # ratio_stats(, None)
    log.logger.info('plot_tool')
    df1 = pd.DataFrame(data={'age': [1, 11, 20], 'name': ['jack', 'mick', 'john']})
    df2 = pd.DataFrame(data={'age': [2, 3, 13, 124], 'name': ['jack', 'mick', 'john', 'aaa']})
    df3 = pd.DataFrame(data={'age': [2, 3, 13, 124], 'name': ['jack', 'mick', 'john', 'aaa']})
    df4 = pd.DataFrame(data={'age': [2, 3, 13, 124], 'name': ['jack', 'mick', 'john', 'aaa']})
    # column_compare_subplot2(df1, df2, df3, df4, 'age', '')
    aa = df3.describe(percentiles=[.1, .25, .8, .9])
    print(aa)
    # 调节具体参数
    # bins调节横坐标分区个数，alpha参数用来设置透明度
    # plt.hist(df1['age'], bins=5,  alpha=0.5, histtype='stepfilled',
    #          color='steelblue', edgecolor='none')
