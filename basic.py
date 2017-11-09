 # -*- coding: utf-8 -*-

import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt


def merge_csv(flist, **kwargs):
    """
        合并csv
    """
    return pd.concat([pd.read_csv(f, **kwargs) for f in flist],axis=1)

def psave(data,fname):
    #保存数据
    with open(fname,'wb') as f:
        pickle.dump(data,f)
        
def pload(fname):
    #读取数据
    with open(fname,'rb') as f:
        pickle.load(f)

def foo():
    pass

