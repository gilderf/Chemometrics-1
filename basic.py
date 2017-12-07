# -*- coding: utf-8 -*-

import pandas as pd
import pickle
import numpy as np
from pyteomics import mzxml, auxiliary
import pandas as pd
import numpy as np
import re
import jcamp
PPM = 1e-6

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
        return pickle.load(f)


def avg_mass(mass, delta=20*PPM, min_intensity=0):
    """
    平均色谱图
    :param mass: dataframe columns = ['mz','intensity','rt']
    :param delta:     离子峰容差
    :param min_intensity: 最小峰强度
    :return: 平均色谱图
    """
    mass = mass.sort_values(by='mz')
    mass['cat'] = (mass.mz.diff() > mass.mz*delta).cumsum()
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


def get_real(rt):
    # 获取保留时间数值min
    return rt.real


def rep(c):
    # repmat保留时间，以匹配mz和intensity
    return np.vstack([c[0], np.tile(c[1], len(c[0])), c[2]]).T


def read_mzxml(file_path):
    """
    读取质谱mzxml文件，将其转换为pandas-dataframe,columns = columns=['intensity','rt','mz']
    :param file_path:
    :return: df
    """
    with mzxml.read(file_path) as reader:
        a = [rep([s['intensity array'], get_real(s['retentionTime']), s['m/z array']]) for s in reader]
    b = np.vstack(a)
    df = pd.DataFrame(b, columns=['intensity', 'rt', 'mz'])
    return df


def regstr(text, regexp):
    # 正则匹配子字符串

    m = re.search(regexp, text)
    if m:
        return m.group(0)


def read_dx(dx_file):
    """
    :param dx_file: .dx红外光谱文件
    :return: pd.Series,波数-吸光度
    """
    with open(dx_file) as dx:
        data = jcamp.jcamp_read(dx)
        ir = pd.Series(data['y'], name=data['yunits'], index=data['x'])
        ir.index.name = data['xunits']
    return ir


