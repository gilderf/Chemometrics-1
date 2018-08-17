import pandas as pd
import xarray as xr
import numpy as np
from pyteomics.mass.mass import isotopic_composition_abundance, isotopologues, nist_mass, Composition
from pyteomics.mass import mass
import heapq
import re
# 常数
PPM = 1e-6


def e_formula(str_formula):
    """
    将formula表示为元素数目， 而不是其他代码
    :param str_formula:
    :return:
    """
    any_digit = any(c.isdigit() for c in str_formula)
    if not any_digit:
        str_formula += '1'
    return str_formula


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
    rt = group.apply(lambda x: x.rt.dot(x.intensity / x.intensity.sum())) # 保留时间加权平均
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


def _test_nearsum():
    ds = xr.open_dataset('./data/mass/1-10.cdf')
    nearsum(ds)

# todo alignment COW and DWT


def molecule_isodist(formula):
    """
    分子同位素文库分布
    """
    all_comp = ((composition2formula(iso_comp),
                 iso_comp,
                 mass.calculate_mass(iso_comp),
                 isotopic_composition_abundance(iso_comp))
                for iso_comp in isotopologues(composition2formula(formula), overall_threshold=1e-4))
    iso_comps = pd.DataFrame(all_comp,
                             columns=['Formula', 'Composition', 'Mass', 'relative_abundance'])
    return iso_comps


def composition2formula(composition):
    """
    组成转换为字符formula
    """
    formula_str = ''
    for element in composition:
        formula_str += element + str(composition[element])
    return formula_str


def test_molecule_isodist():
    moles = pd.read_excel('./data/TW80分子式.xlsx')
    next(molecule_isodist(f) for f in moles.to_dict(orient='records'))


def topk_dict(_dict, k=5):
    """
    _dict中key最大的k个
    """
    dict_ = {m: _dict[m] for m in
             heapq.nlargest(k, _dict, key=_dict.get)}
    return dict_


def filter_nist(isotopes, threshold=1e-5):
    """
    过滤prob小于threshold（1e-4）的元素同位素
    """

    _isotopes = {isotopes[m] for m in isotopes if m > 0 and isotopes[m][1] > threshold}
    return _isotopes


def iso_dist(composition, k=100, ethresshold=1e-5):
    """
    计算同位素分布
    """
    _nist_mass = {el: filter_nist(nist_mass[el], threshold=ethresshold)
                  for el in nist_mass if el in composition}
    # state: value = {mass: prob}
    new_states = {0: 1}
    for e in composition:  # 元素组
        isotopes = _nist_mass[e]
        for ei in range(composition[e]):  # 元素组内
            # 开始状态转移
            if k is not None:
                states = topk_dict(new_states, k=k)  # todo abundance filter, prob small first, fail fast
            else:
                states = new_states
            new_states = {}
            for state in states:
                for _mass, _prob in isotopes:
                    if state + _mass in new_states:
                        # todo cluster
                        new_states[state + _mass] += states[state] * _prob  # 状态转移
                    else:
                        new_states[state + _mass] = states[state] * _prob  # 状态转移
    states = topk_dict(new_states, k=k)
    return states


def test_iso_dist():
    _composition = {"H": 2, "O": 1}
    iso_dist(_composition)
