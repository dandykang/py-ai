#! /usr/bin/env python
# -*- coding:utf8 -*-
from common_tool import ploter
import os
import json

class ComparePer():
    def __init__(self):
        self.test = 'aaa'

    def parse_file(self):
        offline_ratio = {}
        offline_fenwei = {}
        whitefox_ratio = {}
        whitefox_fenwei = {}
        file1 = 'data\\valid_flow_2020-11-26.txt'
        file2 = 'data\\whitefox-1126.txt'
        f1 = open(file1, 'r', encoding='utf-8')
        for line in f1:
            if line.startswith('为0的ratio'):
                tmp1 = line.strip().split(',')
                ratio = tmp1[0].split(':')[1]
                key = tmp1[1].split(':')[0]
                cnt = tmp1[1].split(':')[1]
                if key in offline_ratio:
                    continue
                offline_ratio[key] = (ratio, cnt)
            elif '的10~90分位值' in line:
                tmp2 = line.strip().split(':')
                key = tmp2[0]
                fenwei = tmp2[2]
                if key in offline_fenwei:
                    continue
                fenwei = fenwei.replace('\'', '')
                fenwei_list = json.loads(fenwei)
                offline_fenwei[key] = list(map(self.safe_float, fenwei_list))
        f1.close()

        f2 = open(file2, 'r', encoding='utf-8')
        for line in f2:
            if '@|@' in line:
                featurelist = line.strip().split('@|@')
                for feature in featurelist:
                    tmp3 = feature.strip().split('@')
                    if len(tmp3) < 5:
                        print('feature len < 5, %s' %feature)
                        continue
                    key = tmp3[0]
                    total = tmp3[1]
                    cnt0 = tmp3[2]
                    ratio0 = tmp3[3]
                    fenwei = tmp3[4].split('\u0002')
                    whitefox_ratio[key] = (ratio0, cnt0)
                    whitefox_fenwei[key] = list(map(self.safe_float, fenwei))
        f2.close()

        return offline_ratio, offline_fenwei, whitefox_ratio, whitefox_fenwei

    def safe_float(self, number):
        if number == '\\N':
            return number
        fnum = float(number)
        return round(fnum, 4)

if __name__ == '__main__':
    compare = ComparePer()
    offline_ratio,offline_fenwei,whitefox_ratio ,whitefox_fenwei \
        = compare.parse_file()
    print(offline_ratio)
    print(offline_fenwei)
    print(whitefox_ratio)
    print(whitefox_fenwei)
    comlist = ['web_ch_5day_top60min_every_day_ratio', 'web_ch_30min_ua_top5_rt', 'web_ch_charge_ipTopRatio_day', 'web_ch_charge_phone_provT1Ratio_day', 'web_ch_charge_ctime_ret1_ratio_day', 'web_ch_30min_click_pt_ct', 'web_ch_charge_clickT10Ratio_day', 'web_ch_charge_phone_avgct_day', 'web_ch_10min_suc_rt', 'web_ch_charge_dxRatio_day', 'web_ch_charge_uaT5Ratio_day', 'web_ch_30min_ct', 'web_ch_charge_vsRatio_month', 'web_ch_charge_ipFromNum_day', 'web_ip_10min_channel_ct', 'web_ch_30min_phone_provice_ct', 'web_ch_30min_phone_ew_ct', 'web_ch_charge_count_week', 'web_ch_charge_ctime_ret3_ratio_day', 'web_ch_10min_ct', 'web_ch_charge_elseWhereRatio_day', 'web_ch_charge_clickT10Ratio_month', 'web_ch_charge_ctime_ret5_ratio_day', 'web_ch_30min_ip_provice_ct', 'web_ch_charge_clickRatio_day', 'web_ch_30min_ua_norepeat_rt', 'web_ch_30min_click_pt_top5_rt']
    # comlist = ['web_ch_5day_top60min_every_day_ratio', 'web_ch_30min_ua_top5_rt','web_ch_charge_ipTopRatio_day', 'web_ch_charge_phone_provT1Ratio_day',]
    for f in comlist:
        offline_y = offline_fenwei[f]
        whitefox_y = whitefox_fenwei[f]
        if len(whitefox_y) < 9:
            whitefox_y = [-1,-1,-1,-1,-1,-1,-1,-1,-1]
        x = [1,2,3,4,5,6,7,8,9]
        ylist = [offline_y, whitefox_y]
        ydesc = ['offline_y', 'whitefox_y']
        ploter.chart_line(f, '分位', '分位值', x, ylist, ydesc)
