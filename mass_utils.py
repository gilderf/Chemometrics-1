import pandas as pd

# 常数constant
ppm = 1e-6


def avg_mass(mass, delta=20*ppm, min_intensity=0):
    """
    平均质谱图
    :param mass: dataframe columns = ['mz','intensity','rt']
    :param delta:     离子峰容差 e.g .01 Da
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
