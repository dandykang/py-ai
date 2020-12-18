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


def chart_line(title, xlabel, ylabel, x, ylist, ydeslist, savepath=None):
    plt.title(title)  # 折线图标题
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置
    plt.xlabel(xlabel)  # x轴标题
    plt.ylabel(ylabel)  # y轴标题
    for (y, desc) in zip(ylist, ydeslist):
        plt.plot(x, y, marker='o', markersize=3, label=desc)
        for a, b in zip(x, y):
            plt.text(a, b, b, ha='center', va='bottom', fontsize=10)  # 设置数据标签位置及大小
    plt.legend()  # 让图例生效
    # plt.show()  # 显示折线图
    plt.savefig('data\\' + title + '_compare.png')
    plt.close()

if __name__ == '__main__':
    # ploter = Ploter('多个折线图', '分位值', '分位')
    # ploter = Ploter('chart', 'x', 'y')
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    y1 = [1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9]
    y2 = [1.7, 2.5, 3.1, 4.5, 5.9, 6.4, 7.2, 8.1]
    ylist = [y1, y2]
    desclist = ['y1描述', 'y2描述']
    chart_line('多个折线图', '分位值', '分位', x, ylist, desclist)
