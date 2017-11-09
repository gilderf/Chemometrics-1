# -*- coding: utf-8 -*-

import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt


def merge_csv(flist, **kwargs):
    """
        合并csv
    """
    return pd.concat([pd.read_csv(f, **kwargs) for f in flist], axis=1)


def psave(data, file_name):
    # 保存数据
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)


def pload(file_name):
    # 读取数据
    with open(file_name, 'rb') as f:
        pickle.load(f)


def avg_mass(mass, delta=.1, min_intensity=0.05):
    """
    平均色谱图
    :param mass: dataframe columns = ['mz','intensity','rt']
    :param delta:     离子峰容差
    :param min_intensity: 最小峰强度
    :return: 平均色谱图
    """
    mass.sort_values(by='mz', inplace=True)
    mass['cat'] = (mass.mz.diff() > delta).cumsum()
    group = mass.groupby('cat')
    mz = group.apply(lambda x: x.mz.dot(x.intensity / x.intensity.sum()))
    mz.name = 'mz'
    intensity = group['intensity'].mean()
    avg = pd.concat([mz, intensity], axis=1)
    min_mask = avg.intensity / avg.intensity.max() > min_intensity  # 小峰过滤
    avg = avg.loc[min_mask]
    return avg


def rm_isotopes(mass, delta=1.1):
    """
    去除平均质谱的同位素峰
    :param mass: 平均质谱图 dataframe columns = ['mz','intensity']
    :param delta: 同位素质量差
    :return: 去除同位素后的质谱图
    """
    mass['cat1'] = (mass['mz'].diff() > delta).cumsum()
    groups = mass.groupby('cat1').agg('idxmax')
    mass_rm_isotopes = mass.loc[groups.intensity, ['mz', 'intensity']]
    return mass_rm_isotopes



