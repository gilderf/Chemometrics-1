import jcamp
import spc
from io import StringIO
from pyteomics import mzxml
import pandas as pd
import numpy as np


def read_dx(dx_file):
    """
    jcamp
    :param dx_file: .dx红外光谱文件，或者是.JDX
    :return: pd.Series,波数-吸光度
    """
    with open(dx_file) as dx:
        data = jcamp.jcamp_read(dx)
        ir = pd.Series(data['y'], name=data['yunits'], index=data['x'])
        ir.index.name = data['xunits']
    return ir


def read_spc(spc_file):
    """
    调用spc库，读取spc文件
    :param spc_file:
    :return: dataframe
             index：波数
             values: log(1/R)
    """
    f = spc.File(spc_file)
    stringIO = StringIO(f.data_txt())
    df = pd.read_csv(stringIO, '\t', index_col=0, header=None)
    df1 = pd.DataFrame(df.values, index=df.index, columns=[f.__dict__['ylabel']])
    df1.index.name = f.__dict__['xlabel']
    return df1


def read_hplc_csv(hplc_csv):
    """
    读取csv格式的HPLC数据
    :param hplc_csv:
    :return:
    """
    with open(hplc_csv, 'rb') as csv:
        hplc = pd.read_csv(csv,
                           header=None,
                           names=['retension_time', 'intensity']).set_index('retension_time')
        return hplc


def read_mzxml(file_path):
    """
    读取质谱mzxml文件，将其转换为pandas-dataframe,columns = columns=['intensity','rt','mz']
    调用pyteomics
    :param file_path:
    :return: df
    """
    with mzxml.read(file_path) as reader:
        a = [rep([s['intensity array'], get_real(s['retentionTime']), s['m/z array']]) for s in reader]
    b = np.vstack(a)
    df = pd.DataFrame(b, columns=['intensity', 'rt', 'mz'])
    return df


def rep(c):
    # repmat保留时间，以匹配mz和intensity
    return np.vstack([c[0], np.tile(c[1], len(c[0])), c[2]]).T


def get_real(rt):
    # 获取保留时间数值min
    return rt.real

