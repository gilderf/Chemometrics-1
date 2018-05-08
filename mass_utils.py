import pandas as pd
import xarray as xr
import numpy as np

# 常数
PPM = 1e-6


def avg_iron(mass, delta=20*PPM, min_intensity=0):
    """
    平均离子色谱图 iron chromagram
    :param mass:  dataframe columns = ['mz','intensity','rt']
    :param delta: 离子峰容差 e.g 20PPM
    :param min_intensity: 最小峰强度
    :return: 平均质谱图
    """

    mass = mass.sort_values(by='mz')
    mass['cat'] = (mass.mz.diff() > mass.mz*delta).cumsum()
    group = mass.groupby('cat')
    mz = group.apply(lambda x: x.mz.dot(x.intensity / x.intensity.sum()))
    mz.name = 'mz'
    rt = group.apply(lambda x: x.rt.dot(x.intensity / x.intensity.sum()))
    rt.name = 'rt'
    intensity = group['intensity'].mean()
    avg = pd.concat([mz, intensity, rt], axis=1)
    min_mask = avg.intensity / avg.intensity.max() > min_intensity  # 小峰过滤
    avg = avg.loc[min_mask]
    return avg


def avg_spectrum(mz, intensity, threshold=20*PPM, embeding_avg='avg'):
    """
    avg mass spectrum，合并很接近的mz, 总强度
    :param mz: 1d numpy array
    :param intensity: 1d numpy array
    :threshold:
    :return:
    """
    df = pd.DataFrame({'mz': mz,
                       'ints': intensity})
    df.sort_values(by='mz', inplace=True)
    abs_threshold = df.mz * threshold
    mask = np.cumsum(df.mz.diff() > abs_threshold)  # mz组的间隔
    grps = df.groupby(mask)
    ints = grps.ints.sum()
    if embeding_avg == 'avg':
        ints.index = grps.mz.mean()
    else:
        ints.index = grps.apply(lambda grp: grp.mz.dot((grp.ints / grp.ints.sum())))  # weighted mean 比较费时间
    # ints = ints[ints > 0]  # pandas masking
    return ints


def test_avg_spectrum():
    mz = [100, 100.01]
    ints = [1, 2]
    threshold = .1
    avg_spectrum(mz, ints, threshold)


def nearsum(ds, threshold=.1):
    """
    对整个扫描时间段(多scans)的m/z和rt都相近的进行合并
    m/z相近： delta范围内
    rt相近：  同一个scan
    合并m/z：组内简单平均，加权平均pandas的apply(func)很慢
    :return: [scan_id, m/z, sum_intensity]
    """

    scans = np.zeros_like(ds.mass_values, dtype=np.int32)
    scans[ds.scan_index] = 1
    scans = scans.cumsum()
    df = pd.DataFrame({'scans': scans,
                       'mz': ds.mass_values.values,
                       'its': ds.intensity_values.values})
    grps = df.groupby(np.cumsum(df.mz.diff() > threshold))
    _nearsum = pd.concat([grps.scans.first(), grps.mz.mean(), grps.its.sum()], axis=1)
    return _nearsum


def test_nearsum():
    ds = xr.open_dataset('./data/mass/1-10.cdf')
    nearsum(ds)


# todo alignment
