#! /usr/bin/env python
# -*- coding:utf8 -*-
from common_tool import config_manager as cfg

white_product_list = cfg.get_config('channel', 'white_product_list').split(',')
white_channel_list = cfg.get_config('channel', 'white_channel_list').split(',')
black_channel_list = cfg.get_config('channel', 'black_channel_list').split(',')
phone_list = cfg.get_config('channel', 'phone_list').split(',')
phone_list = list(set(phone_list))
print('white_product_list size', len(white_product_list))
print('white_channel_list size', len(white_channel_list))
print('black_channel_list size', len(black_channel_list))
print('phone_list size', len(phone_list))

def judge_channel_type(channel_code):
    channel_code = str(channel_code)
    if (channel_code in white_channel_list):
        return '0'
    elif (channel_code in black_channel_list):
        return '1'
    else:
        return 'NAN'

def judge_type_by_pdandcn(product_code, channel_code, phone):
    channel_code = str(channel_code)
    product_code = str(product_code)
    phone = str(phone)
    if (channel_code in white_channel_list or product_code in white_product_list):
        return '0'
    elif phone in phone_list:
        return '0'
    elif (channel_code in black_channel_list):
        return '1'
    else:
        return 'NAN'


if __name__ == '__main__':
    print(white_channel_list)
    print(black_channel_list)
    print(judge_type_by_pdandcn('6980390341000002451','5300002891','13400681011'))