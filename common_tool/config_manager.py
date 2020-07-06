#! /usr/bin/env python
# -*- coding:utf8 -*-
from configparser import ConfigParser


def get_config(section, key):
    cp = ConfigParser()
    # 以.ini结尾的配置文件
    cp.read("config.ini")

    # 获取mysql中的host值
    v = cp.get(section, key)
    return v


if __name__ == '__main__':
    sample_path = get_config('file', 'sample_path')
    print(sample_path)
