
# coding: utf-8
from pyteomics import mzxml, auxiliary
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import read_mzxml, pload
import pickle
import re
import sys
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# datafile = "./data010.mzXML"  # 质谱数据
# sample_name = re.sub('[\./]', '', datafile)[:-5]
iron_lib_file = 'irons.p'  # 一级质谱库
irons = pload(iron_lib_file)
f_rts_range = 'rts_range.p'
rts_range = pload(f_rts_range)
#
mzB = 1331.8279
ppm = 1e-6
error_threshold = 20*ppm
rtB = 10.91  # B组理论保留时间


def rt_real(df, rtrange=(8.500, 10.987), mz=1331.8279, et=20*ppm):
    #  实测保留时间计算
    dfB = df.loc[df.rt.between(*rtrange)]#选取大范围的样本点
    dfB = dfB.loc[abs(dfB.mz-mz)<mz*et]#选取mzB
    grpB = dfB.groupby('mz')
    dfB1 = dfB.loc[grpB['intensity'].idxmax()]#eic顶点
    rtB_real = dfB1.rt.dot(dfB1.intensity/dfB1.intensity.sum())#顶点rt强度加权评价
    return rtB_real


def rt_correction(df, rtB=10.91, *args, **kwargs):
    #  保留时间校正
    #     mz = 1331.8279
    #     ppm = 1e-6
    #     error_threshold = mz*20*ppm
    #  1. 大范围取点。 rt范围：8.500-10.987min， mz范围：mz+-error_threshold
    #  2. 取各mz的EIC顶点(rt,intensity)
    #  3. 求各mz顶点强度intensity加权保留时间rt - rt_real
    rtB_real = rt_real(df, *args, **kwargs)
    delta = rtB-rtB_real
    if abs(delta) < .1:
        return df
    elif abs(delta) >= .1:
        df['rt'] = df.rt+delta
        return df
    elif abs(delta) >= .5:
        raise ValueError("delta超过了0.5")


def read_and_clean(datafile, irons):
    # 读取mzxml,保留5位小数
    df = read_mzxml(datafile)
    df = df.round(5).astype(np.float32)
    # m/z 初筛
    mzs = pd.concat([irons.mz1, irons.mz2], axis=0).reset_index(drop=True).astype(np.float32)
    mask_mas = np.any(abs(df.mz.values.reshape(-1, 1) - mzs.values) < .1, axis=1)
    df = df.loc[mask_mas]
    # 保留时间校正
    df = rt_correction(df, rtB)
    return df


def avg_mass(mass, delta=.01, min_intensity=0):
    """
    平均质谱图
    :param mass: dataframe columns = ['mz','intensity','rt']
    :param delta:     离子峰容差 e.g .01 Da
    :param min_intensity: 最小峰强度
    :return: 平均质谱图
    """
    mass = mass.sort_values(by='mz')
    mass['cat'] = (mass.mz.diff() > delta).cumsum()
    group = mass.groupby('cat')
    mz = group.apply(lambda x: x.mz.dot(x.intensity / x.intensity.sum()))
    mz.name = 'mz'
    rt = group.apply(lambda x: x.rt.dot(x.intensity / x.intensity.sum()))
    rt.name = 'rt'
    intensity = group['intensity'].mean()
    avg = pd.concat([mz, intensity,rt], axis=1)
    min_mask = avg.intensity / avg.intensity.max() > min_intensity  # 小峰过滤
    avg = avg.loc[min_mask]
    return avg


def matchA(ironsi, avg, delta1=20*ppm):
    # 匹配特征离子
    ironsi1_v = ironsi.mz1.values
    ironsi2_v = ironsi.mz2.values
    mzs = avg.mz.values
    r1 = abs(ironsi1_v-mzs.reshape(-1,1))<ironsi1_v*delta1
    r2 = abs(ironsi2_v-mzs.reshape(-1,1))<ironsi2_v*delta1
    r = r1|r2
    r1 = r.dot(ironsi.n)
    r2 = pd.Series(r1,index=avg.index)
    r2.loc[~r.any(axis=1)] = -9999
    avg.loc[:,'n']= r2
    return avg


def matchB_G(ironsi,avg,delta1 = 20*ppm,delta2=.01):
    ironsi1_v = ironsi.mz1.values
    ironsi2_v = ironsi.mz2.values
    ironsi_rtv = ironsi.rt.values
    mzs = avg.mz.values
    rts = avg.rt.values
    r1 = abs(ironsi1_v-mzs.reshape(-1,1))<ironsi1_v*delta1
    r2 = abs(ironsi2_v-mzs.reshape(-1,1))<ironsi2_v*delta1
    r3 = abs(ironsi_rtv-rts.reshape(-1,1))<ironsi_rtv*delta2#保留时间
    #r3 = abs(ironsi_rtv-rts.reshape(-1,1))<delta2#保留时间-绝对
    r = (r1 | r2) & r3
    r1 = r.dot(ironsi.n)
    r2 = pd.Series(r1,index=avg.index)
    r2.loc[~r.any(axis=1)] = -9999
    avg.loc[:,'n']= r2
    return avg


def match(ironsi,avg,cat,delta1 = 20*ppm,delta2=.01):
    if cat in ['A-1','A-2']:
        return matchA(ironsi,avg,delta1)
    elif cat in ['B', 'C', 'D', 'E', 'F', 'G']:
        return matchB_G(ironsi,avg,delta1,delta2)
    else:
        raise('没有这个组！')


def match_df(df, irons, rts_range):
    # 1级质谱离子库匹配
    avgs = []
    for cat in rts_range.index.tolist():
        idf = irons.loc[irons.cat == cat]
        rtrange = rts_range.loc[cat]
        avg = avg_mass(df.loc[df.rt.between(*rtrange)])
        avg.loc[:, 'grp'] = cat
        avgs.append(match(idf, avg, cat))
    avg_dfs = pd.concat(avgs)
    avg_dfs.loc[avg_dfs.n == -9999, 'n'] = np.nan
    return avg_dfs


def result2excel(out, filename, out_type='zh'):
    # 输出结果一级质谱库匹配结果到excel
    # 1. 匹配结果
    # 2. 结构式图
    if out_type == 'zh':
        out.columns = [u'm/z', u'intensity', u'保留时间(min)', u'组分名称', u'聚合度(n)']
    elif out_type == 'en':
        ['m/z (MS)', 'Intensity (MS)', 'Retention Time (min)', 'Group', 'Degree of Polymerization(n=w+x+y+z)']
    out.loc[:, out.columns[1]] = out.iloc[:, 1].astype(np.float64)
    out.loc[:, out.columns[-1]] = out.loc[:, out.columns[-1]].astype(np.int)
    out = out.round({out.columns[0]: 4, out.columns[1]: 2, out.columns[2]: 3})
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    out.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    fig_position = out.reset_index(drop=True).groupby(out.columns[3])[out.columns[-1]].idxmin() + 2
    for grp in fig_position.index.tolist():
        worksheet.insert_image('F' + str(fig_position[grp]), grp + '.png')
    writer.save()


def do_plot(out, filename, figsize=(10, 5)):
    cmaps = iter(['Greys', 'Reds', 'Blues', 'Greens', 'Oranges', 'Purples',
                  'cool', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                  'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn'])
    filled_markers = iter(('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'))
    for grp in out.组分名称.unique():
        d = out.loc[out.组分名称 == grp]
        plt.rcParams['figure.figsize'] = figsize
        plt.scatter(d.loc[:, '保留时间(min)'].values, d.loc[:, 'm/z'].values,
                    cmap=next(cmaps),
                    marker=next(filled_markers),
                    s=np.sqrt(d.intensity),
                    label=grp,
                    alpha=.7,
                    edgecolors=None)
    plt.legend(bbox_to_anchor=(1.1, 1), loc="upper right")
    plt.savefig(filename, dpi=300)


def tw80(datafile, out_file=None, out_type='zh'):
    df = read_and_clean(datafile, irons)
    avg_dfs = match_df(df, irons, rts_range)
    out = avg_dfs.loc[~avg_dfs.n.isnull()].sort_values(['grp', 'n'])
    if out_file:
        result2excel(out, out_file, out_type=out_type)  # 结果输出
    return out


if __name__ == '__main__':
    datafile = "./data050.mzXML"  # 质谱数据
    sample_name = re.sub('[\./]', '', datafile)[:-5]
    tw80(datafile, out_file='test_py'+sample_name+'xlsx', out_type='zh')
    print(sample_name)

