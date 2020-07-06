#! /usr/bin/env python
# -*- coding:utf8 -*-
import pandas as pd
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
print(os.path.abspath(__file__))
print(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)
print(sys.path)
from common_tool import config_manager as cfg
from feature_project import channel_info
from common_tool import logger
import time



# np.set_printoptions(threshold=np.inf)  # 加上这一句
# 导入后加入以下列，再显示时显示完全。
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class ProcessFlow:
    def __init__(self):
        self.m = ''


sample_path = cfg.get_config('file', 'sample_path')
files = os.listdir(sample_path)
data_column = cfg.get_config('sample', 'all_column').split(',')
result_path = cfg.get_config('file', 'sample_result')
result_column = cfg.get_config('sample', 'keep_column')
log_path = cfg.get_config('file', 'log_path')
log = logger.Logger(log_path+'process.log', level='debug', when='H')



def load_data_by_dir():
    chunks = []
    for root, dirs, files in os.walk(sample_path, topdown=False):
        for dir in dirs:
            sub_path = os.path.join(root, dir)
            files = os.listdir(sub_path)
            files_csv = list(filter(lambda x: x[-4:] == '.csv', files))
            for file in files_csv:
                full_path_file = os.path.join(sub_path, file)
                iter_tmp = pd.read_csv(full_path_file, sep='@', header=None, names=data_column,
                                       iterator=True, encoding='utf-8', dtype={'phone': object},low_memory=False)
                loop = True
                chunksize = 100000
                while loop:
                    try:
                        chunk = iter_tmp.get_chunk(chunksize)
                        log.logger.info('读取的文件%s, chunk条目%s' % (full_path_file, str(chunk.shape)))
                        # chunk = chunk[chunk.apply(lambda row: channel_info.judge_type_by_pdandcn(row['x_in_productcode'], row['x_in_channelcode']), axis=1)  != 'NAN']
                        # chunk['tmp_flag'] = chunk.apply(lambda row: channel_info.judge_type_by_pdandcn(row['x_in_productcode'], row['x_in_channelcode']), axis=1)
                        # chunk = chunk[chunk['tmp_flag'] != 'NAN']
                        chunk = add_classify_label(chunk)
                        log.logger.info('读取的文件%s, chunk经渠道过滤后条目%s' % (full_path_file, str(chunk.shape)))
                        chunks.append(chunk)
                    except StopIteration:
                        loop = False
                        log.logger.info("Iteration is stopped.")
    sample_data = pd.concat(chunks, ignore_index=True)

    log.logger.info('========load_data end========data row num : %d' % len(sample_data))
    return sample_data

def load_data():
    files_csv = list(filter(lambda x: x[-4:] == '.csv', files))
    # note: 普通方式加载小数据量的文件
    # data_list = []
    # for file in files_csv:
    #     tmp = pd.read_csv(sample_path + file, sep='@_', header=None, names=data_column, encoding='utf-8')
    #     data_list.append(tmp)
    # sample_data = pd.concat(data_list)

    chunks = []
    for file in files_csv:
        iter_tmp = pd.read_csv(sample_path + file, sep='@', header=None, names=data_column,
                          iterator=True, encoding='utf-8')
        loop = True
        chunksize = 100000
        while loop:
            try:
                chunk = iter_tmp.get_chunk(chunksize)
                log.logger.info('读取的文件%s, chunk条目%s' %(file,str(chunk.shape)))
                # chunk = chunk[chunk.apply(lambda row: channel_info.judge_type_by_pdandcn(row['x_in_productcode'], row['x_in_channelcode']), axis=1)  != 'NAN']
                # chunk['tmp_flag'] = chunk.apply(lambda row: channel_info.judge_type_by_pdandcn(row['x_in_productcode'], row['x_in_channelcode']), axis=1)
                # chunk = chunk[chunk['tmp_flag'] != 'NAN']
                chunk = add_classify_label(chunk)
                log.logger.info('读取的文件%s, chunk经渠道过滤后条目%s' %(file,str(chunk.shape)))
                chunks.append(chunk)
            except StopIteration:
                loop = False
                log.logger.info("Iteration is stopped.")

    sample_data = pd.concat(chunks, ignore_index=True)

    log.logger.info('========load_data end========data row num : %d' % len(sample_data))
    return sample_data


# 打标签，划分正负样本，根据渠道标记进行划分
def add_classify_label(input_data):
    # input_data['y_label'] = input_data['x_in_channelcode'] .apply(lambda x: channel_info.judge_channel_type(x))
    input_data['y_label'] = input_data.apply(lambda row: channel_info.judge_type_by_pdandcn(row['x_in_productcode'], row['x_in_channelcode'], row['phone']), axis=1)
    effective_data = input_data[input_data['y_label'].apply(lambda x: x != 'NAN')]
    log.logger.info('========add_classify_label end========effective data row num : %d' % len(effective_data))
    return effective_data


# 数据处理，包括：
# 1、基于行为的特征补充
# 2、无效标签的过滤
def feature_generate(input_df):
    # 判断归属地是否一致
    # input_df['cur_same_prov'] = input_df.apply(lambda row: row['x_ext6'] == row['x_in_ext21'], 1, 0).astype(float)
    input_df['cur_same_prov'] = input_df.apply(lambda row: check_same(row['x_ext6'], row['x_in_ext21']), axis=1).astype(int)
    log.logger.info('========feature_generate cur_same_prov end========')

    input_df['charge_wait_time'] = input_df['x_in_commlog20'].apply(lambda x: time_interval(x))
    log.logger.info('========feature_generate charge_wait_time end========')

    input_df['cur_vcode_time_max'] = input_df['x_in_commlog26'].apply(lambda x: check_input_time(x, 'max'))
    log.logger.info('========feature_generate cur_vcode_time_max end========')
    input_df['cur_vcode_time_min'] = input_df['x_in_commlog26'].apply(lambda x: check_input_time(x, 'min'))
    log.logger.info('========feature_generate cur_vcode_time_min end========')
    input_df['cur_vcode_time_mean'] = input_df['x_in_commlog26'].apply(lambda x: check_input_time(x, 'mean'))
    log.logger.info('========feature_generate cur_vcode_time_mean end========')
    input_df['cur_vcode_time_sum'] = input_df['x_in_commlog26'].apply(lambda x: check_input_time(x, 'sum'))
    log.logger.info('========feature_generate cur_vcode_time_sum end========')

    input_df['cur_dot_ct'] = input_df['x_in_commlog30'].apply(lambda x: array_count(x, ','))
    log.logger.info('========feature_generate cur_dot_ct end========')

    input_df['cur_xff_ct'] = input_df['x_in_ext26'].apply(lambda x: array_count(x, ','))
    log.logger.info('========feature_generate cur_xff_ct end========')

    input_df['cur_ipisp'] = input_df['x_ext10'].apply(lambda x: check_ipisp(x))
    log.logger.info('========feature_generate cur_ipisp end========')

    log.logger.info('========all feature_generate end========effective data row num : %d' % len(input_df))

    trans_list = ['x_in_chargepolicy','x_in_commlog5','x_ext17','x_in_commlog29','x_ext16','x_ext6']
    for col in trans_list:
        input_df[col] = input_df[col].apply(lambda x: check_nan_trans(x, col))

    return input_df

def check_nan_trans(value, col_name):
    if pd.isna(value):
        if col_name == 'x_ext6':
            return 0
        else:
            return -1
    else:
        return value

# 筛选要保留的字段
def filter_feature(df):
    keep_column_list = result_column.split(',')
    return df[keep_column_list]

# 保存样本处理的结果
def save_result(df):
    file_data = time.strftime('%Y-%m-%d_%H%M%S', time.localtime(time.time()))
    result_file = result_path + file_data + '.csv'
    df.to_csv(result_file, header=True, index=False)
    log.logger.info('========save_result end, path : %s========' % result_file)
    log.logger.info('========df y_label = 1 count : %d' % len(df[df['y_label'] == '1']))
    log.logger.info('========df y_label = 0 count : %d' % len(df[df['y_label'] == '0']))

    log.logger.info('========df summary : \n%s' % df.describe())

def check_ipisp(str):
    if str == '电信':
        return 1
    elif str == '移动':
        return 2
    elif str == '联通':
        return 3
    elif str == '铁通':
        return 4
    else:
        return 5


def array_count(str_in, split_str):
    str_in = str(str_in)
    if len(str_in) > 0:
        str_list = str_in.split(split_str)
        return len(str_list)
    else:
        return 0


def time_interval(time_str):
    time_str = str(time_str)
    if len(time_str) > 0:
        time_list = time_str.split(',')
        if (len(time_list) != 2):
            return -1
        try:
            timeArrayStart = time.strptime(time_list[0], "%Y%m%d%H%M%S%f")
            timeArrayEnd = time.strptime(time_list[1], "%Y%m%d%H%M%S%f")
            timeStamp1 = int(time.mktime(timeArrayStart))
            timeStamp2 = int(time.mktime(timeArrayEnd))
            return timeStamp2 - timeStamp1
        except Exception as e:
            log.logger.error('str(Exception):\t', str(Exception))
            log.logger.error('str(e):\t\t', str(e))
            log.logger.error('str(time_str):\t\t', str(time_str))
            return -1
    else:
        return -1


def check_same(v1, v2):
    if v1 == v2:
        return True
    else:
        return False


def check_input_time(input_time, type):
    if pd.isna(input_time):
        return -1

    input_time = str(input_time)
    if len(input_time) > 0:
        time_list = input_time.split(',')
        time_list = list(map(float, time_list))
        if type == 'max':
            return max(time_list)
        elif type == 'min':
            return min(time_list)
        elif type == 'mean':
            return (float)(sum(time_list) / len(time_list))
        elif type == 'sum':
            return sum(time_list)
        else:
            return -1
    else:
        return -1

def data_value_check(input_df):
    keep_column_list = result_column.split(',')
    total_len = len(input_df)
    out_str = '\n总行数：' + str(total_len) + '\n'
    out_col_0_str = '\n全部为0值行：\n'
    for col in keep_column_list:
        tmp_df = input_df[input_df[col] == 0]
        tmp_len = len(tmp_df)
        out_str = out_str + 'ratio:'+str(round(tmp_len/total_len,2)) + ',' + col + ':' + str(tmp_len) + '\n'
        if tmp_len == total_len:
            out_col_0_str = out_col_0_str + col + '\n'
    log.logger.info(out_str)
    log.logger.info(out_col_0_str)
    return out_str

def process():
    sample_data_all = load_data_by_dir()
    log.logger.info('列是否全为空\n %s' %sample_data_all.isnull().all())  # 判断某列是否全部为NaN
    log.logger.info('是否存在空值\n %s' %sample_data_all.isnull().any())  # 判断某列是否有NaN
    log.logger.info(sample_data_all.head(1))
    log.logger.info(sample_data_all.dtypes)
    log.logger.info('\n')

    sample_data_all = feature_generate(sample_data_all)
    log.logger.info(sample_data_all.head(1))
    log.logger.info('\n')
    sample_data_all = filter_feature(sample_data_all)

    log.logger.info('列是否全为空\n %s' %sample_data_all.isnull().all())  # 判断某列是否全部为NaN
    log.logger.info('是否存在空值\n %s' %sample_data_all.isnull().any())  # 判断某列是否有NaN

    data_value_check(sample_data_all)

    save_result(sample_data_all)



if __name__ == '__main__':
    process()
    # load_data_by_dir()
    # charge_wait_time = time_interval('20200610122022133,20200610122055123')
    # log.logger.info(time.strftime('%Y-%m-%d_%H%M%S', time.localtime(time.time())))
