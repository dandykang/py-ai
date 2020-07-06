#! /usr/bin/env python
# -*- coding:utf8 -*-
import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
import math

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
print(os.path.abspath(__file__))
print(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)
print(sys.path)
from common_tool import config_manager as cfg
from common_tool import logger
import time

normal_col = cfg.get_config('data', 'deal_normal')
zscore_col = cfg.get_config('data', 'deal_zscore')
result_column = cfg.get_config('sample', 'keep_column')

log_path = cfg.get_config('file', 'log_path')
log = logger.Logger(log_path + 'normalization.log', level='debug', when='H')
result_path = cfg.get_config('file', 'sample_result')

del_col = []
col_minmax_dic = {}


def load_data_by_dir():
    files = os.listdir(result_path)
    files_csv = list(filter(lambda x: x[-7:] == 'all.csv', files))
    chunks = []
    for file in files_csv:
        iter_tmp = pd.read_csv(result_path + file, iterator=True, encoding='utf-8')
        loop = True
        chunksize = 50000
        while loop:
            try:
                chunk = iter_tmp.get_chunk(chunksize)
                log.logger.info('读取的文件%s, chunk条目%s' % (file, str(chunk.shape)))
                chunks.append(chunk)
            except StopIteration:
                loop = False
                log.logger.info("Iteration is stopped.")
    sample_data = pd.concat(chunks, ignore_index=True)

    log.logger.info('========load_data end========data row num : %d' % len(sample_data))
    return sample_data


def normal_train(df, col_list):
    method = lambda x: round((x - x.min()) / (x.max() - x.min()), 4)
    df_col_list = df.columns.values.tolist()
    for col in col_list:
        if col in df_col_list:
            col_minmax_dic[col] = (df[col].min(), df[col].max())
            df[col] = df[[col]].apply(method)
        else:
            log.logger.info('训练集归一化操作时，读入数据中不包含列col : %s ' % col)

    return df


def nromal_test_func(col, row):
    v = row[col]
    min_v = col_minmax_dic[col][0]
    max_v = col_minmax_dic[col][1]
    if v < min_v:
        v = min_v

    if v > max_v:
        v = max_v

    return round((v - min_v) / (max_v - min_v), 4)


def normal_test(df, col_list):
    df_col_list = df.columns.values.tolist()
    for col in col_list:
        if col in df_col_list:
            df[col] = df.apply(lambda row: nromal_test_func(col, row), axis=1)
        else:
            log.logger.info('测试集归一化操作时，读入数据中不包含列col : %s ' % col)
    return df


def z_score(df, col_list):
    method = lambda x: round((x - np.mean(x)) / np.std(x), 4)
    for col in col_list:
        df[col] = df[[col]].apply(method)

    return df


def data_normalization_train(input_df, input_list=None):
    if input_list is None:
        normal_list = normal_col.split(',')
    else:
        normal_list = input_list

    col_list = list(set(normal_list) - set(del_col))
    df = normal_train(input_df, col_list)
    return df


def data_normalization_test(input_df, input_list=None):
    if input_list is None:
        normal_list = normal_col.split(',')
    else:
        normal_list = input_list

    col_list = list(set(normal_list) - set(del_col))
    df = normal_test(input_df, col_list)
    return df


# 保存样本处理的结果
def save_result(df_train, df_test, file_name):
    file_data = time.strftime('%Y-%m-%d_%H%M%S', time.localtime(time.time()))
    df_train_file = result_path + file_data + file_name + '_train.csv'
    df_test_file = result_path + file_data + file_name + '_test.csv'
    df_train.to_csv(df_train_file, header=True, index=False)
    df_test.to_csv(df_test_file, header=True, index=False)
    log.logger.info('========save_result end, path : %s========' % df_train_file)
    log.logger.info('========save_result end, path : %s========' % df_test_file)


# 保存样本处理的结果
def save_result_one(df, file_name):
    file_data = time.strftime('%Y-%m-%d_%H%M%S', time.localtime(time.time()))
    result_file = result_path + file_data + file_name + '.csv'
    df.to_csv(result_file, header=True, index=False)
    log.logger.info('========save_result end, path : %s========' % result_file)


def random_data(input_df):
    df_0 = input_df[input_df['y_label'] == 0]
    df_1 = input_df[input_df['y_label'] == 1]
    log.logger.info('df_0 shape: %s' % str(df_0.shape))
    log.logger.info('df_1 shape: %s' % str(df_1.shape))
    # replace 是否允许重复取样，即一条数据多次选取。默认为否
    # random_state 指定随机数种子后，每次选取的结果就固定了
    df_1_sample = df_1.sample(100000, replace=False, random_state=3)
    log.logger.info('df_1_sample shape: %s' % str(df_1_sample.shape))

    df_new = pd.concat([df_0, df_1_sample], ignore_index=True)
    log.logger.info('df_new shape: %s' % str(df_new.shape))
    return df_new


def del_all_zero_col(input_df):
    total_len = len(input_df)
    df_col_list = input_df.columns.values.tolist()
    col_list = result_column.split(',')
    for col in col_list:
        if col in df_col_list:
            tmp_df = input_df[input_df[col] == 0]
            tmp_len = len(tmp_df)
            if tmp_len == total_len:
                del_col.append(col)
                # 直接在原df上删除对应列
                del input_df[col]
                log.logger.info('col : %s 全部为0值，删除之' % col)
        else:
            del_col.append(col)
            log.logger.info('读入数据中不包含列col : %s ' % col)
    return input_df


def check_col_all_zero(input_df):
    total_len = len(input_df)
    df_col_list = input_df.columns.values.tolist()
    for col in df_col_list:
        tmp_df = input_df[input_df[col] == 0]
        tmp_len = len(tmp_df)
        if tmp_len == total_len:
            log.logger.info('col : %s 全部为0值，请注意！' % col)


def change_to_zero_func(value):
    if float(value) >= 0:
        return value
    else:
        return 0


def change_to_zero(input_df):
    keep_list = result_column.split(',')
    col_list = list(set(keep_list) - set(del_col))
    for col in col_list:
        input_df[col] = input_df[col].apply(lambda x: change_to_zero_func(x))

    return input_df


def log_func(value):
    if float(value) >= 0:
        # +1 防止原数据是0 不能做变换
        ret_value = math.log(value + 1, 10)
        # log.logger.info('输入值: %s, 变换后：%s' % (str(value), str(ret_value)))
        return ret_value
    else:
        log.logger.error('对数变换输入值异常: %s' % str(value))
        return 0


def data_to_log(input_df, log_col_list):
    df_col_list = input_df.columns.values.tolist()
    for col in log_col_list:
        if col in df_col_list:
            input_df[col] = input_df[col].apply(lambda x: log_func(x))
        else:
            log.logger.info('取对数操作时，读入数据中不包含列col : %s ' % col)

    return input_df


def filter_avilable_data(input_df):
    ret_df = input_df[input_df['x_in_chargepolicy'] != -1]
    return ret_df


def sample_original_data():
    sample_data_all = load_data_by_dir()
    log.logger.info('load_data_by_dir shape: %s' % str(sample_data_all.shape))
    sample_data_all = del_all_zero_col(sample_data_all)
    log.logger.info('del_all_zero_col shape: %s' % str(sample_data_all.shape))
    sample_data_all = filter_avilable_data(sample_data_all)
    log.logger.info('filter_avilable_data shape: %s' % str(sample_data_all.shape))
    # 对正样本取样 合并负样本
    sample_data_all = random_data(sample_data_all)
    log.logger.info('random_data shape: %s' % str(sample_data_all.shape))
    save_result_one(sample_data_all, '_original')
    return sample_data_all


def sample_log_nor_deal(sample_data_all):
    # -1调整为0
    sample_data_all = change_to_zero(sample_data_all)

    # todo 统一对指定列数据做log变换
    log_col_list = ['charge_wait_time', 'web_ch_charge_conversion_ratio_day', 'cur_vcode_time_max',
                    'cur_vcode_time_mean', 'cur_vcode_time_sum', 'web_ch_charge_phoneFromNum_day',
                    'web_ch_charge_phoneFromNum_month', 'web_ch_charge_phoneFromNum_week', 'web_ip_10min_suc_ct',
                    'web_ip_distinct_hour_day', 'web_phone_10min_change_address', 'web_phone_success_count_month',
                    'web_phone_success_count_week', 'web_phone_total_count_day', 'web_phone_total_count_month',
                    'web_phone_total_count_week', 'x_ext6', 'x_ext17', 'x_in_commlog5', 'cur_vcode_time_min',
                    'web_ch_10min_fpay_ct', 'web_ch_charge_ip_avg_charge_day', 'web_ch_charge_ip_avg_charge_week',
                    'web_ip_10min_phone_ct', 'web_ip_distinct_hour_day', 'web_ip_distinct_succ_phone_day',
                    'web_ip_distinct_succ_phone_month', 'web_ip_distinct_succ_phone_week',
                    'web_ip_ply_charge_ratio_month', 'web_ip_succ_distinct_ch_day', 'web_ip_succ_distinct_ch_week',
                    'web_ip_total_count_day', 'web_ip_total_count_month', 'web_ip_total_count_week',
                    'web_phone_10min_charge_ct', 'web_phone_chargetype1_count_month',
                    'web_phone_chargetype1_count_week', 'web_phone_succ_amount_week', 'web_phone_success_count_month',
                    'web_phone_total_count_week', 'x_ext6', 'x_ext17', 'x_in_chargepolicy']
    nor_col_list = ['charge_wait_time', 'x_in_chargepolicy', 'x_in_commlog5', 'cur_vcode_time_max',
                    'cur_vcode_time_min', 'cur_vcode_time_mean', 'cur_vcode_time_sum', 'x_ext17', 'x_in_commlog29',
                    'cur_dot_ct', 'x_ext16', 'cur_xff_ct', 'cur_ipisp', 'x_ext6', 'web_ch_charge_count_day',
                    'web_ch_charge_phoneFromNum_day', 'web_ch_charge_ipFromNum_day', 'web_ch_charge_phone_avgct_day',
                    'web_ch_5day_top60min_every_day_count', 'web_ch_charge_foreignip_count_day',
                    'web_ch_charge_ip_avg_charge_day', 'web_phone_distinct_ua_day', 'web_phone_success_count_day',
                    'web_phone_total_count_day', 'web_phone_chargetype0_count_day', 'web_phone_chargetype1_count_day',
                    'web_phone_succ_amount_day', 'app_phone_success_count_day', 'app_phone_total_count_day',
                    'app_phone_chargetype0_count_day', 'app_phone_chargetype1_count_day', 'app_phone_succ_amount_day',
                    'web_ip_total_count_day', 'web_ip_succ_distinct_ch_day', 'web_ip_distinct_hour_day',
                    'web_ip_distinct_ua_day', 'web_ip_distinct_succ_phone_day', 'app_ip_total_count_day',
                    'app_ip_succ_distinct_ch_day', 'app_ip_distinct_hour_day', 'app_ip_distinct_succ_phone_day',
                    'web_ch_charge_count_week', 'web_ch_charge_phoneFromNum_week', 'web_ch_charge_ipFromNum_week',
                    'web_ch_charge_ip_avg_charge_week', 'web_phone_distinct_ua_week', 'web_phone_success_count_week',
                    'web_phone_total_count_week', 'web_phone_chargetype0_count_week',
                    'web_phone_chargetype1_count_week', 'web_phone_succ_amount_week', 'app_phone_success_count_week',
                    'app_phone_total_count_week', 'app_phone_chargetype0_count_week',
                    'app_phone_chargetype1_count_week', 'app_phone_succ_amount_week', 'web_ip_total_count_week',
                    'web_ip_succ_distinct_ch_week', 'web_ip_distinct_succ_phone_week', 'app_ip_total_count_week',
                    'app_ip_succ_distinct_ch_week', 'app_ip_distinct_succ_phone_week', 'web_ch_charge_count_month',
                    'web_ch_charge_phoneFromNum_month', 'web_ch_charge_ipFromNum_month',
                    'web_ch_charge_phone_avgct_month', 'web_ch_charge_foreignip_count_month',
                    'web_ch_charge_ip_avg_charge_month', 'web_phone_distinct_ua_month', 'web_phone_success_count_month',
                    'web_phone_total_count_month', 'web_phone_chargetype0_count_month',
                    'web_phone_chargetype1_count_month', 'web_phone_succ_amount_month', 'app_phone_total_count_month',
                    'app_phone_chargetype0_count_month', 'app_phone_chargetype1_count_month',
                    'app_phone_succ_amount_month', 'web_ip_total_count_month', 'web_ip_succ_distinct_ch_month',
                    'web_ip_distinct_succ_phone_month', 'app_ip_total_count_month', 'app_ip_succ_distinct_ch_month',
                    'app_ip_distinct_succ_phone_month', 'web_ip_10min_suc_ct', 'web_ip_10min_channel_ct',
                    'web_ip_10min_phone_ct', 'web_ip_1h_ua_ct', 'app_ip_10min_suc_ct', 'web_ch_10min_ct',
                    'web_ch_30min_ct', 'web_ch_10min_fpay_ct', 'web_ch_30min_phone_ew_ct',
                    'web_ch_30min_phone_provice_ct', 'web_ch_30min_ip_provice_ct', 'web_ch_30min_click_pt_ct',
                    'web_phone_10min_charge_ct', 'web_phone_10min_change_address', 'app_phone_10min_charge_ct']

    sample_data_all = data_to_log(sample_data_all, log_col_list)
    log.logger.info('完成对数变换！')
    check_col_all_zero(sample_data_all)
    log.logger.info('完成对数变换后全0的检查！')
    # 分样本为为验证集和测试集
    # 直接对data进行测试集、训练集划分
    sample_data_train, sample_data_test = train_test_split(sample_data_all, test_size=0.20, random_state=2)
    train_0 = sample_data_train[sample_data_train['y_label'] == 0]
    train_1 = sample_data_train[sample_data_train['y_label'] == 1]
    test_0 = sample_data_test[sample_data_test['y_label'] == 0]
    test_1 = sample_data_test[sample_data_test['y_label'] == 1]
    log.logger.info('train_0 shape: %s, train_1 shape: %s' % (str(train_0.shape), str(train_1.shape)))
    log.logger.info('test_0 shape: %s, test_1 shape: %s' % (str(test_0.shape), str(test_1.shape)))

    # 数据归一化处理,单独对训练集做归一化，记录最大最小值
    sample_data_train = data_normalization_train(sample_data_train, nor_col_list)
    log.logger.info('col_minmax_dic: %s' % str(col_minmax_dic))
    log.logger.info('完成训练集归一化！')
    check_col_all_zero(sample_data_train)
    log.logger.info('完成训练集归一化后全0的检查！')

    sample_data_test = data_normalization_test(sample_data_test, nor_col_list)
    log.logger.info('完成测试集归一化！')
    check_col_all_zero(sample_data_test)
    log.logger.info('完成测试集归一化后全0的检查！')

    log.logger.info('sample_data_train_normalization shape: %s' % str(sample_data_train.shape))
    log.logger.info('sample_data_test_normalization shape: %s' % str(sample_data_test.shape))
    save_result(sample_data_train, sample_data_test, 'log_normal')


def deal_log_nor():
    sample_data_all = load_data_by_dir()
    log.logger.info('load_data_by_dir shape: %s' % str(sample_data_all.shape))
    sample_data_all = del_all_zero_col(sample_data_all)
    log.logger.info('del_all_zero_col shape: %s' % str(sample_data_all.shape))
    # 对正样本取样 合并负样本
    sample_data_all = random_data(sample_data_all)
    log.logger.info('random_data shape: %s' % str(sample_data_all.shape))

    # -1调整为0
    sample_data_all = change_to_zero(sample_data_all)

    # todo 可加一下只保留需要的特征名，去掉app的特征，注意保留 y_label
    feature_00to02_list = ['app_ip_distinct_succ_phone_day', 'app_phone_chargetype0_count_week',
                           'app_ip_succ_distinct_ch_month', 'web_ch_charge_foreignip_count_month',
                           'app_phone_chargetype1_count_day', 'app_phone_success_count_week',
                           'web_ch_charge_phoneFromNum_month', 'charge_wait_time', 'app_phone_chargetype1_count_month',
                           'web_ip_10min_suc_ct', 'web_ch_charge_phone_provT1Ratio_month',
                           'web_phone_total_count_month', 'web_phone_success_count_week',
                           'web_ch_charge_phoneFromNum_day', 'web_ch_charge_foreignip_count_day',
                           'app_ip_total_count_day', 'web_ch_charge_clickT10Ratio_month', 'cur_vcode_time_max',
                           'app_phone_succ_amount_week', 'app_ip_total_count_week', 'web_phone_total_count_day',
                           'web_ch_charge_phoneFromNum_week', 'x_ext6', 'web_phone_10min_change_address',
                           'web_ch_charge_ltRatio_day', 'web_ch_charge_phone_provT1Ratio_day',
                           'app_phone_total_count_week', 'web_phone_success_count_month', 'x_ext17',
                           'web_phone_10min_charge_ct', 'web_ch_charge_ltRatio_month', 'app_ip_succ_distinct_ch_day',
                           'app_ip_distinct_succ_phone_month', 'web_ch_charge_clickRatio_month', 'x_in_commlog5',
                           'web_ch_charge_clickT10Ratio_day', 'app_phone_succ_amount_day',
                           'app_phone_success_count_day', 'app_ip_distinct_succ_phone_week',
                           'web_ch_charge_uaRatio_day', 'web_ch_30min_ua_norepeat_rt', 'app_ip_charge_succ_ratio_day',
                           'app_phone_chargetype0_count_month', 'app_phone_total_count_day', 'web_ip_distinct_hour_day',
                           'app_ip_succ_distinct_ch_week', 'cur_vcode_time_sum', 'web_phone_total_count_week',
                           'app_ip_total_count_month', 'app_phone_succ_amount_month', 'cur_vcode_time_mean',
                           'app_phone_chargetype1_count_week', 'app_phone_total_count_month',
                           'web_ch_charge_uaRatio_month', 'app_ip_distinct_hour_day', 'web_ip_total_count_day',
                           'web_phone_chargetype1_count_month', 'web_ip_distinct_succ_phone_month',
                           'web_ip_total_count_week', 'x_in_chargepolicy', 'web_ip_total_count_month',
                           'app_ip_charge_succ_ratio_week', 'web_ch_charge_across_city_ratio_day', 'cur_vcode_time_min',
                           'web_ip_10min_phone_ct', 'web_ip_succ_distinct_ch_week', 'web_phone_succ_amount_week',
                           'app_ip_total_count_week', 'web_ip_ply_charge_ratio_month', 'web_ip_distinct_succ_phone_day',
                           'x_ext6', 'web_ch_charge_ip_avg_charge_week', 'web_ch_charge_ltRatio_day',
                           'web_ch_10min_fpay_ct', 'app_phone_total_count_week', 'web_phone_success_count_month',
                           'web_ip_succ_distinct_ch_day', 'web_phone_10min_charge_ct', 'x_ext17',
                           'web_ch_charge_ip_avg_charge_day', 'web_ch_charge_conversion_ratio_day',
                           'web_phone_chargetype1_count_week', 'web_ch_charge_clickT10Ratio_day',
                           'web_ch_charge_uaRatio_day', 'app_ip_charge_succ_ratio_day',
                           'app_phone_chargetype0_count_month', 'web_ip_distinct_hour_day',
                           'app_ip_succ_distinct_ch_week', 'web_phone_total_count_week',
                           'web_ip_distinct_succ_phone_week', 'app_phone_succ_amount_month',
                           'app_phone_total_count_month']
    feature_00to02_list = list(set(feature_00to02_list))
    del_app_list = []
    for f in feature_00to02_list:
        if f.startswith('app_'):
            del_app_list.append(f)
    feature_00to02_list = list(set(feature_00to02_list) - set(del_app_list))
    feature_00to02_list.append('y_label')
    log.logger.info('00to02的特征数量%d，\n特征内容%s ' % (len(feature_00to02_list), str(feature_00to02_list)))
    sample_data_all = sample_data_all[feature_00to02_list]
    log.logger.info('filter 00to02 feature shape: %s' % str(sample_data_all.shape))
    # todo 统一对指定列数据做log变换
    log_col_list = ['charge_wait_time', 'web_ch_charge_conversion_ratio_day', 'cur_vcode_time_max',
                    'cur_vcode_time_mean', 'cur_vcode_time_sum', 'web_ch_charge_phoneFromNum_day',
                    'web_ch_charge_phoneFromNum_month', 'web_ch_charge_phoneFromNum_week', 'web_ip_10min_suc_ct',
                    'web_ip_distinct_hour_day', 'web_phone_10min_change_address', 'web_phone_success_count_month',
                    'web_phone_success_count_week', 'web_phone_total_count_day', 'web_phone_total_count_month',
                    'web_phone_total_count_week', 'x_ext6', 'x_ext17', 'x_in_commlog5', 'cur_vcode_time_min',
                    'web_ch_10min_fpay_ct', 'web_ch_charge_ip_avg_charge_day', 'web_ch_charge_ip_avg_charge_week',
                    'web_ip_10min_phone_ct', 'web_ip_distinct_hour_day', 'web_ip_distinct_succ_phone_day',
                    'web_ip_distinct_succ_phone_month', 'web_ip_distinct_succ_phone_week',
                    'web_ip_ply_charge_ratio_month', 'web_ip_succ_distinct_ch_day', 'web_ip_succ_distinct_ch_week',
                    'web_ip_total_count_day', 'web_ip_total_count_month', 'web_ip_total_count_week',
                    'web_phone_10min_charge_ct', 'web_phone_chargetype1_count_month',
                    'web_phone_chargetype1_count_week', 'web_phone_succ_amount_week', 'web_phone_success_count_month',
                    'web_phone_total_count_week', 'x_ext6', 'x_ext17', 'x_in_chargepolicy']
    sample_data_all = data_to_log(sample_data_all, log_col_list)
    log.logger.info('完成对数变换！')
    check_col_all_zero(sample_data_all)
    log.logger.info('完成对数变换后全0的检查！')
    # 分样本为为验证集和测试集
    # 直接对data进行测试集、训练集划分
    sample_data_train, sample_data_test = train_test_split(sample_data_all, test_size=0.20, random_state=2)
    train_0 = sample_data_train[sample_data_train['y_label'] == 0]
    train_1 = sample_data_train[sample_data_train['y_label'] == 1]
    test_0 = sample_data_test[sample_data_test['y_label'] == 0]
    test_1 = sample_data_test[sample_data_test['y_label'] == 1]
    log.logger.info('train_0 shape: %s, train_1 shape: %s' % (str(train_0.shape), str(train_1.shape)))
    log.logger.info('test_0 shape: %s, test_1 shape: %s' % (str(test_0.shape), str(test_1.shape)))

    save_result(sample_data_train, sample_data_test, 'original')

    # 数据归一化处理,单独对训练集做归一化，记录最大最小值
    sample_data_train = data_normalization_train(sample_data_train, log_col_list)
    log.logger.info('col_minmax_dic: %s' % str(col_minmax_dic))
    log.logger.info('完成训练集归一化！')
    check_col_all_zero(sample_data_train)
    log.logger.info('完成训练集归一化后全0的检查！')

    sample_data_test = data_normalization_test(sample_data_test, log_col_list)
    log.logger.info('完成测试集归一化！')
    check_col_all_zero(sample_data_test)
    log.logger.info('完成测试集归一化后全0的检查！')

    log.logger.info('sample_data_train_normalization shape: %s' % str(sample_data_train.shape))
    log.logger.info('sample_data_test_normalization shape: %s' % str(sample_data_test.shape))
    save_result(sample_data_train, sample_data_test, 'log_normal')


if __name__ == '__main__':
    df = sample_original_data()
    sample_log_nor_deal(df)
    # deal_log_nor()
    # 定义一个数据框
    # df = pd.DataFrame(data={'age': [18, 19, 20],'name': ['jack', 'mick', 'john']})
    # log.logger.info('df_new shape: %s' % str(df.shape))
    # log.logger.info('df age max: %d' % df['age'].max())
    # method = lambda x: round((x - x.min()) / (x.max() - x.min()), 4)
    # df['age'] = df[['age']].apply(method)
    # print(df)
    #
    # normal_train(df, ['age'])
    # log.logger.info('col_minmax_dic: %s' % str(col_minmax_dic))
    # df2 = pd.DataFrame(data={'age': [100, 19.5, 1000, 22, -2], 'name': ['jack', 'mick', 'john', 'abcd', 'efg']})
    # tmp_df2 = data_to_log(df2, ['age'])
    # print(tmp_df2)
    # df2_nor = normal_test(df2, ['age'])
    # print(df2_nor)
    # 字典操作
    # a = {'score': (99, 98)}
    # print(a['score'][1])
