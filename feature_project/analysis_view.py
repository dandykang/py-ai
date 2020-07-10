#! /usr/bin/env python
# -*- coding:utf8 -*-
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
from feature_project import plot_tool

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
log = logger.Logger(log_path + 'analysis_view.log', level='debug', when='H')
result_path = cfg.get_config('file', 'sample_result')
result_column = cfg.get_config('sample', 'keep_column')


def load_all_data_by_dir():
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


def load_sample_data_by_file_name(file_name):
    files = os.listdir(result_path)
    files_csv = list(filter(lambda x: x == file_name, files))
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

    log.logger.info('========load_data[file_name:%s] end========data row num : %d' % (file_name,len(sample_data)))
    return sample_data

# 标签分类比例
def view_sample_ratio(input_df, result_label):
    label_0, label_1 = plot_tool.ratio_stats(input_df, result_label)
    log.logger.info('训练集数据中，label_0 数量为：%i,label_1 数量为：%i,label_0 占比例为：%.2f%%'
                    % (label_0, label_1, (label_0 / (label_0 + label_1) * 100)))
# 样本各特征分布情况
def view_sample_column_desc(input_df):
    col_list = result_column.split(',')
    for col in col_list:
        plot_tool.column_describe(input_df, col)

# 正负样本各特征对比
def view_sample_column_compare(sample_data_ori, sample_data_nor, pic_name):
    col_list = result_column.split(',')
    plot_tool.column_compare_subplot(sample_data_ori, sample_data_nor, col_list, 'y_label', pic_name)

# 正负样本部分征对比
def view_sample_part_column_compare(sample_data_ori, sample_data_nor, col_list, pic_name):
    plot_tool.column_compare_subplot(sample_data_ori, sample_data_nor, col_list, 'y_label', pic_name)


fetrures_X_06 = ['web_ch_5day_top60min_every_day_count','web_phone_ip_diff_prov_ratio_day','web_phone_ip_diff_prov_ratio_week','web_ch_charge_wapRatio_month','web_ch_charge_elseWhereRatio_day','web_ip_phone_diff_prov_ratio_day','web_ch_charge_elseWhereRatio_week','web_ip_phone_diff_prov_ratio_week','web_ch_charge_elseWhereRatio_month','web_phone_ip_diff_prov_ratio_month','web_ip_phone_diff_prov_ratio_month','x_in_commlog29','cur_dot_ct','x_in_commlog5','web_ch_charge_count_month','web_ch_charge_vsRatio_month','web_ch_30min_phone_ew_ct']
fetrures_X_07 = ['web_ch_charge_dxRatio_day','web_ch_charge_ydRatio_month','web_ch_charge_dxRatio_month','web_ch_30min_ua_top5_rt','web_ch_charge_count_day','web_ch_charge_ydRatio_day','web_ch_charge_count_week','web_ch_10min_ct','web_ch_30min_ct']
fetrures_X_08 = ['x_ext16']
fetrures_X_05 = ['x_in_commlog29','cur_dot_ct','x_ext16','cur_ipisp','web_ch_charge_count_day','web_ch_charge_elseWhereRatio_day','web_ch_charge_ipTopRatio_day','web_ch_charge_ydRatio_day','web_ch_charge_dxRatio_day','web_ch_charge_phone_avgct_day','web_ch_5day_top60min_every_day_count','web_ch_charge_selfnet_ratio_day','web_ch_charge_across_city_ratio_day','web_phone_chargetype0_count_day','web_phone_ip_diff_prov_ratio_day','web_ip_phone_diff_prov_ratio_day','web_ch_charge_count_week','web_ch_charge_elseWhereRatio_week','web_ch_charge_suc_ratio_week','web_phone_chargetype0_count_week','web_phone_ip_diff_prov_ratio_week','web_ip_phone_diff_prov_ratio_week','web_ch_charge_count_month','web_ch_charge_elseWhereRatio_month','web_ch_charge_vsRatio_month','web_ch_charge_ydRatio_month','web_ch_charge_dxRatio_month','web_ch_charge_wapRatio_month','web_ch_charge_selfnet_ratio_month','web_phone_distinct_ua_month','web_phone_chargetype0_count_month','web_phone_ip_diff_prov_ratio_month','web_ip_phone_diff_prov_ratio_month','web_ip_10min_phone_ct','web_ch_10min_ct','web_ch_30min_ct','web_ch_10min_suc_rt','web_ch_30min_phone_ew_ct','web_ch_30min_ua_top5_rt','x_ext16','cur_ipisp','web_ch_charge_count_day','web_ch_charge_elseWhereRatio_day','web_ch_charge_ipTopRatio_day','web_ch_charge_ydRatio_day','web_ch_charge_dxRatio_day','web_ch_charge_selfnet_ratio_day','web_phone_ip_diff_prov_ratio_day','web_ip_phone_diff_prov_ratio_day','web_ch_charge_count_week','web_ch_charge_elseWhereRatio_week','web_ch_charge_suc_ratio_week','web_phone_ip_diff_prov_ratio_week','web_ip_phone_diff_prov_ratio_week','web_ch_charge_elseWhereRatio_month','web_ch_charge_ipFromNum_month','web_ch_charge_vsRatio_month','web_ch_charge_ydRatio_month','web_ch_charge_dxRatio_month','web_ch_charge_selfnet_ratio_month','web_phone_ip_diff_prov_ratio_month','web_ip_phone_diff_prov_ratio_month','web_ch_30min_ua_top5_rt','x_in_commlog29','cur_dot_ct','x_ext16','cur_ipisp','web_ch_charge_count_day','web_ch_charge_elseWhereRatio_day','web_ch_charge_ydRatio_day','web_ch_charge_dxRatio_day','web_ch_5day_top60min_every_day_count','web_ch_charge_across_city_ratio_day','web_phone_chargetype0_count_day','web_phone_ip_diff_prov_ratio_day','web_ip_phone_diff_prov_ratio_day','web_ch_charge_count_week','web_ch_charge_elseWhereRatio_week','web_phone_chargetype0_count_week','web_phone_ip_diff_prov_ratio_week','web_ip_phone_diff_prov_ratio_week','web_ch_charge_count_month','web_ch_charge_elseWhereRatio_month','web_ch_charge_vsRatio_month','web_ch_charge_ydRatio_month','web_ch_charge_dxRatio_month','web_ch_charge_wapRatio_month','web_phone_chargetype0_count_month','web_phone_ip_diff_prov_ratio_month','web_ch_10min_ct','web_ch_30min_ct','web_ch_30min_phone_ew_ct','web_ch_30min_ua_top5_rt']
fetrures_X_00to01 = ['web_ch_charge_uaRatio_month','web_ch_charge_phone_provT1Ratio_month','web_ch_charge_foreignip_count_month','app_phone_chargetype1_count_month','web_ip_10min_suc_ct','web_ch_30min_ua_norepeat_rt','app_phone_chargetype1_count_day','app_phone_succ_amount_day','web_ip_distinct_hour_day','app_ip_total_count_day','app_ip_succ_distinct_ch_day','app_ip_charge_succ_ratio_day','app_ip_distinct_succ_phone_day','web_ch_charge_phoneFromNum_week','web_phone_success_count_week','app_phone_success_count_week','app_phone_chargetype0_count_week','app_phone_chargetype1_count_week','app_phone_succ_amount_week','web_ch_charge_phoneFromNum_month','web_ch_charge_clickT10Ratio_month','charge_wait_time','x_in_commlog5','cur_vcode_time_max','cur_vcode_time_mean','cur_vcode_time_sum','web_ch_charge_phoneFromNum_day','web_ch_charge_clickT10Ratio_day','web_ch_charge_uaRatio_day','web_ch_charge_phone_provT1Ratio_day','web_ch_charge_foreignip_count_day','web_phone_total_count_day','app_phone_success_count_day','app_phone_total_count_day','web_ch_charge_uaRatio_month','web_ch_charge_ltRatio_month','web_ch_charge_phone_provT1Ratio_month','web_ch_charge_foreignip_count_month','web_phone_success_count_month','web_phone_total_count_month','app_phone_total_count_month','app_phone_chargetype0_count_month','app_phone_chargetype1_count_month','app_phone_succ_amount_month','app_ip_total_count_month','app_ip_succ_distinct_ch_month','app_ip_distinct_succ_phone_month','web_ip_10min_suc_ct','web_phone_10min_charge_ct','web_phone_10min_change_address','app_phone_chargetype1_count_day','app_phone_succ_amount_day','app_ip_total_count_day','app_ip_succ_distinct_ch_day','app_ip_distinct_succ_phone_day','web_phone_success_count_week','web_phone_total_count_week','app_phone_success_count_week','app_phone_total_count_week','app_phone_chargetype0_count_week','app_phone_chargetype1_count_week','app_phone_succ_amount_week','app_ip_total_count_week','app_ip_succ_distinct_ch_week','app_ip_distinct_succ_phone_week','web_ch_charge_clickRatio_month','charge_wait_time','x_in_commlog5','cur_vcode_time_max','cur_vcode_time_mean','cur_vcode_time_sum','x_ext17','x_ext6','web_ch_charge_phoneFromNum_day','web_ch_charge_ltRatio_day','web_ch_charge_phone_provT1Ratio_day','web_ch_charge_foreignip_count_day','web_phone_total_count_day','app_phone_success_count_day','app_phone_total_count_day','web_ch_charge_uaRatio_month','web_ch_charge_phone_provT1Ratio_month','web_ch_charge_foreignip_count_month','app_phone_chargetype1_count_month','web_ip_10min_suc_ct','web_ch_30min_ua_norepeat_rt','app_phone_chargetype1_count_day','app_phone_succ_amount_day','web_ip_distinct_hour_day','app_ip_total_count_day','app_ip_succ_distinct_ch_day','app_ip_charge_succ_ratio_day','app_ip_distinct_succ_phone_day','web_ch_charge_phoneFromNum_week','web_phone_success_count_week','app_phone_success_count_week','app_phone_chargetype0_count_week','app_phone_chargetype1_count_week','app_phone_succ_amount_week','web_ch_charge_phoneFromNum_month','web_ch_charge_clickT10Ratio_month','charge_wait_time','x_in_commlog5','cur_vcode_time_max','cur_vcode_time_mean','cur_vcode_time_sum','web_ch_charge_phoneFromNum_day','web_ch_charge_clickT10Ratio_day','web_ch_charge_uaRatio_day','web_ch_charge_phone_provT1Ratio_day','web_ch_charge_foreignip_count_day','web_phone_total_count_day','app_phone_success_count_day','app_phone_total_count_day']
fetrures_X_01to02 = ['web_phone_success_count_month','app_phone_total_count_month','app_phone_chargetype0_count_month','app_phone_succ_amount_month','web_ip_total_count_month','web_phone_10min_charge_ct','web_ip_total_count_day','web_ip_succ_distinct_ch_day','web_ip_distinct_succ_phone_day','app_ip_distinct_hour_day','web_ch_charge_ip_avg_charge_week','web_phone_total_count_week','app_phone_total_count_week','web_ip_total_count_week','web_ip_succ_distinct_ch_week','web_ip_distinct_succ_phone_week','app_ip_total_count_week','app_ip_succ_distinct_ch_week','app_ip_charge_succ_ratio_week','x_in_chargepolicy','cur_vcode_time_min','x_ext17','x_ext6','web_ch_charge_ltRatio_day','web_ch_charge_ip_avg_charge_day','web_phone_chargetype1_count_month','web_ip_ply_charge_ratio_month','web_ip_10min_phone_ct','web_ch_10min_fpay_ct','web_ip_distinct_hour_day','app_ip_distinct_hour_day','app_ip_charge_succ_ratio_day','web_phone_chargetype1_count_week','web_phone_succ_amount_week','web_ch_charge_clickT10Ratio_day','web_ch_charge_uaRatio_day','web_ch_charge_across_city_ratio_day','web_ch_charge_conversion_ratio_day','x_in_chargepolicy','cur_vcode_time_min','x_ext17','x_ext6','web_ch_charge_ltRatio_day','web_ch_charge_ip_avg_charge_day','web_ip_total_count_day','web_ip_succ_distinct_ch_day','web_ip_distinct_succ_phone_day','app_ip_distinct_hour_day','web_ch_charge_ip_avg_charge_week','web_phone_total_count_week','app_phone_total_count_week','web_ip_total_count_week','web_ip_succ_distinct_ch_week','web_ip_distinct_succ_phone_week','app_ip_total_count_week','app_ip_succ_distinct_ch_week','app_ip_charge_succ_ratio_week','web_phone_success_count_month','app_phone_total_count_month','app_phone_chargetype0_count_month','app_phone_succ_amount_month','web_ip_total_count_month','web_ip_distinct_succ_phone_month','web_phone_10min_charge_ct']
fetrures_X_05to07 = list(set(fetrures_X_05) - set(fetrures_X_06) - set(fetrures_X_07))
fetrures_X_07to08 = list(set(fetrures_X_07) | set(fetrures_X_08))
fetrures_X_00to01 = list(set(fetrures_X_00to01))
fetrures_X_01to02 = list(set(fetrures_X_01to02))
log.logger.info('fetrures_X_00to01数量%d, fetrures_X_01to02数量%d, fetrures_X_07to08数量%d'
                % (len(fetrures_X_00to01), len(fetrures_X_01to02), len(fetrures_X_07to08)))
fetrures_X_random = ['web_phone_10min_charge_ct', 'app_ip_charge_succ_ratio_day', 'web_phone_10min_change_address', 'web_ch_charge_ltRatio_day', 'app_ip_total_count_day', 'x_in_commlog5', 'web_ch_charge_ip_avg_charge_day', 'web_ch_charge_uaRatio_month', 'app_ip_succ_distinct_ch_week', 'app_ip_distinct_succ_phone_week', 'x_ext17', 'web_ch_charge_phoneFromNum_week', 'web_ip_total_count_day', 'web_ip_succ_distinct_ch_week', 'web_ch_charge_phoneFromNum_month']
fetrures_X_lt02 = ['web_ch_30min_ua_norepeat_rt', 'web_ip_10min_suc_ct', 'web_ch_charge_ip_avg_charge_day', 'web_phone_chargetype1_count_week', 'web_ip_distinct_succ_phone_week', 'cur_vcode_time_sum', 'web_ch_charge_uaRatio_day', 'x_ext6', 'web_ip_total_count_week', 'charge_wait_time', 'web_phone_success_count_month', 'web_ch_charge_foreignip_count_day', 'web_phone_chargetype1_count_month', 'web_ip_ply_charge_ratio_month', 'web_ip_distinct_succ_phone_day', 'web_phone_succ_amount_week', 'web_ch_charge_ltRatio_month', 'web_phone_total_count_day', 'web_phone_total_count_week', 'cur_vcode_time_min', 'web_ch_10min_fpay_ct', 'web_ch_charge_ip_avg_charge_week', 'web_ch_charge_uaRatio_month', 'web_ch_charge_phone_provT1Ratio_day', 'cur_vcode_time_mean', 'web_ip_total_count_month', 'web_ip_succ_distinct_ch_day', 'web_ch_charge_conversion_ratio_day', 'web_ch_charge_phoneFromNum_week', 'x_ext17', 'x_in_commlog5', 'web_ch_charge_clickT10Ratio_month', 'web_ch_charge_clickT10Ratio_day', 'web_ip_distinct_hour_day', 'web_phone_10min_charge_ct', 'web_ch_charge_clickRatio_month', 'web_ip_distinct_succ_phone_month', 'web_ch_charge_phone_provT1Ratio_month', 'web_phone_10min_change_address', 'web_ch_charge_foreignip_count_month', 'web_ip_total_count_day', 'web_ip_10min_phone_ct', 'web_ip_succ_distinct_ch_week', 'x_in_chargepolicy', 'cur_vcode_time_max', 'web_ch_charge_phoneFromNum_day', 'web_ch_charge_across_city_ratio_day', 'web_ch_charge_ltRatio_day', 'web_phone_success_count_week', 'web_phone_total_count_month', 'web_ch_charge_phoneFromNum_month']
fetrures_all = ['cur_same_prov','charge_wait_time','x_in_chargepolicy','x_in_commlog5','cur_vcode_time_max','cur_vcode_time_min','cur_vcode_time_mean','cur_vcode_time_sum','x_ext17','x_in_commlog29','cur_dot_ct','x_ext16','cur_xff_ct','cur_ipisp','x_ext6','web_ch_charge_count_day','web_ch_charge_elseWhereRatio_day','web_ch_charge_phoneFromNum_day','web_ch_charge_ipFromNum_day','web_ch_charge_ipTopRatio_day','web_ch_charge_clickT10Ratio_day','web_ch_charge_clickRatio_day','web_ch_charge_vsRatio_day','wseb_ch_charge_uaT5Ratio_day','web_ch_charge_uaRatio_day','web_ch_charge_ctime_ret1_ratio_day','web_ch_charge_ctime_ret3_ratio_day','web_ch_charge_ctime_ret5_ratio_day','web_ch_charge_ydRatio_day','web_ch_charge_ltRatio_day','web_ch_charge_dxRatio_day','web_ch_charge_wapRatio_day','web_ch_charge_phone_avgct_day','web_ch_5day_top60min_every_day_count','web_ch_5day_top60min_every_day_ratio','web_ch_charge_phone_provT1Ratio_day','web_ch_charge_foreignip_count_day','web_ch_charge_selfnet_ratio_day','web_ch_charge_across_city_ratio_day','web_ch_charge_fpay_ratio_day','web_ch_charge_conversion_ratio_day','web_ch_charge_ip_avg_charge_day','web_phone_distinct_ua_day','web_phone_success_count_day','web_phone_total_count_day','web_phone_chargetype0_count_day','web_phone_chargetype1_count_day','web_phone_succ_amount_day','web_phone_ip_diff_prov_ratio_day','app_phone_success_count_day','app_phone_total_count_day','app_phone_chargetype0_count_day','app_phone_chargetype1_count_day','app_phone_succ_amount_day','web_ip_total_count_day','web_ip_succ_distinct_ch_day','web_ip_phone_diff_prov_ratio_day','web_ip_distinct_hour_day','web_ip_distinct_ua_day','web_ip_ply_charge_ratio_day','web_ip_charge_succ_ratio_day','web_ip_distinct_succ_phone_day','app_ip_total_count_day','app_ip_succ_distinct_ch_day','app_ip_distinct_hour_day','app_ip_charge_succ_ratio_day','app_ip_distinct_succ_phone_day','web_ch_charge_count_week','web_ch_charge_elseWhereRatio_week','web_ch_charge_phoneFromNum_week','web_ch_charge_ipFromNum_week','web_ch_charge_suc_ratio_week','web_ch_charge_ip_avg_charge_week','web_phone_distinct_ua_week','web_phone_success_count_week','web_phone_total_count_week','web_phone_chargetype0_count_week','web_phone_chargetype1_count_week','web_phone_succ_amount_week','web_phone_ip_diff_prov_ratio_week','app_phone_success_count_week','app_phone_total_count_week','app_phone_chargetype0_count_week','app_phone_chargetype1_count_week','app_phone_succ_amount_week','web_ip_total_count_week','web_ip_succ_distinct_ch_week','web_ip_phone_diff_prov_ratio_week','web_ip_ply_charge_ratio_week','web_ip_charge_succ_ratio_week','web_ip_distinct_succ_phone_week','app_ip_total_count_week','app_ip_succ_distinct_ch_week','app_ip_charge_succ_ratio_week','app_ip_distinct_succ_phone_week','web_ch_charge_count_month','web_ch_charge_elseWhereRatio_month','web_ch_charge_phoneFromNum_month','web_ch_charge_ipFromNum_month','web_ch_charge_ipTopRatio_month','web_ch_charge_clickT10Ratio_month','web_ch_charge_clickRatio_month','web_ch_charge_vsRatio_month','web_ch_charge_uaT5Ratio_month','web_ch_charge_uaRatio_month','web_ch_charge_ydRatio_month','web_ch_charge_ltRatio_month','web_ch_charge_dxRatio_month','web_ch_charge_wapRatio_month','web_ch_charge_phone_avgct_month','web_ch_charge_phone_provT1Ratio_month','web_ch_charge_foreignip_count_month','web_ch_charge_selfnet_ratio_month','web_ch_charge_ip_avg_charge_month','web_phone_distinct_ua_month','web_phone_success_count_month','web_phone_total_count_month','web_phone_chargetype0_count_month','web_phone_chargetype1_count_month','web_phone_succ_amount_month','web_phone_ip_diff_prov_ratio_month','app_phone_total_count_month','app_phone_chargetype0_count_month','app_phone_chargetype1_count_month','app_phone_succ_amount_month','web_ip_total_count_month','web_ip_succ_distinct_ch_month','web_ip_phone_diff_prov_ratio_month','web_ip_ply_charge_ratio_month','web_ip_charge_succ_ratio_month','web_ip_distinct_succ_phone_month','app_ip_total_count_month','app_ip_succ_distinct_ch_month','app_ip_charge_succ_ratio_month','app_ip_distinct_succ_phone_month','web_ip_10min_suc_ct','web_ip_10min_channel_ct','web_ip_10min_phone_ct','web_ip_1h_ua_ct','web_ch_10min_ct','web_ch_30min_ct','web_ch_10min_suc_rt','web_ch_10min_fpay_ct','web_ch_10min_fpay_rt','web_ch_30min_phone_ew_ct','web_ch_30min_phone_provice_ct','web_ch_30min_ip_provice_ct','web_ch_30min_click_pt_top5_rt','web_ch_30min_click_pt_ct','web_ch_30min_ua_top5_rt','web_ch_30min_ua_norepeat_rt','web_phone_10min_charge_ct','web_phone_10min_change_address','app_phone_10min_charge_ct']

log.logger.info('fetrures_X_00to01信息：%s' %str(fetrures_X_00to01))
log.logger.info('fetrures_X_01to02信息：%s' %str(fetrures_X_01to02))

def analysis():
    sample_data_ori = load_sample_data_by_file_name('2020-07-02_102117_original.csv')
    # sample_data_ori_2 = load_sample_data_by_file_name('2020-06-30_105904_ori_train.csv')
    # sample_data_ori = pd.concat([sample_data_ori_1,sample_data_ori_2])
    sample_data_nor_1 = load_sample_data_by_file_name('2020-07-02_102303log_normal_train.csv')
    sample_data_nor_2 = load_sample_data_by_file_name('2020-07-02_102303log_normal_test.csv')
    sample_data_nor = pd.concat([sample_data_nor_1,sample_data_nor_2])

    # 查看样本选取的比例
    view_sample_ratio(sample_data_ori, 'y_label')
    view_sample_ratio(sample_data_nor, 'y_label')

    # 比较全部样本特征
    # view_sample_column_compare(sample_data_ori, sample_data_nor, 'all_col')
    # view_sample_part_column_compare(sample_data_ori, sample_data_nor,fetrures_X_00to01, '00to01')
    # view_sample_part_column_compare(sample_data_ori, sample_data_nor,fetrures_X_01to02, '01to02')
    # view_sample_part_column_compare(sample_data_ori, sample_data_nor,fetrures_X_07to08, '07to08')
    view_sample_part_column_compare(sample_data_ori, sample_data_nor,fetrures_all, 'test')




def heatmap_view():
    sample_data_nor = load_sample_data_by_file_name('2020-06-24_104945_narmal_0.csv')
    # sample_data_nor = load_sample_data_by_file_name('2020-06-24_093545_narmal_ori.csv')
    df_col_list = sample_data_nor.columns.values.tolist()
    df_col_num = len(df_col_list)
    method = 'spearman'
    # 建立共线性表格
    # DataFrame.corr(method='pearson', min_periods=1)
    # 参数说明：
    # method：可选值为{‘pearson’, ‘kendall’, ‘spearman’}
    # pearson：Pearson相关系数来衡量两个数据集合是否在一条线上面，即针对线性数据的相关系数计算，针对非线性数据便会有误差。
    # kendall：用于反映分类变量相关性的指标，即针对无序序列的相关系数，非正太分布的数据
    # spearman：非线性的，非正太分析的数据的相关系数
    # min_periods：样本最少的数据量
    # 返回值：各类型之间的相关系数DataFrame表格。
    i = 0
    while i < df_col_num:
        start = i
        end = i + 52
        if end > df_col_num:
            end = df_col_num
        sub_sample_data_nor = sample_data_nor.iloc[:, start:end]
        sub_df_col_list = sub_sample_data_nor.columns.values.tolist()
        if 'y_label' not in sub_df_col_list:
            sub_sample_data_nor['y_label'] = sample_data_nor['y_label']

        correlation_table = pd.DataFrame(sub_sample_data_nor.corr(method= method))
        df_tmp = correlation_table[abs(correlation_table['y_label']) < 0.1]
        # df_tmp = df_tmp[abs(df_tmp['y_label']) > 0.1]
        # print(df_tmp['y_label'])
        print('\n结果：', df_tmp.iloc[:-1, -1])
        # 热力图
        # sns.heatmap(correlation_table,annot=True)
        i = end
        # plt.xticks(fontsize=8)  # 对坐标的值数值，大小限制
        # plt.yticks(fontsize=8)  # 对坐标的值数值，大小限制
        # plt.savefig('sample_pic\\relative\\' + method + '_' + str(i) + 'relative' + '.png')
        # plt.show()




if __name__ == '__main__':
    analysis()
    # heatmap_view()
    # data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # df = pd.DataFrame(data)
    # print(df)
    # print('===========')
    # df_1 = df.iloc[:,[0,1]]
    # print(df_1)
    # print('===========')
    # df_2 = df.iloc[:,2:3]
    # print(df_2)
    # sns.heatmap(data, annot=True)
    # plt.yticks(rotation=120)
    # plt.show()