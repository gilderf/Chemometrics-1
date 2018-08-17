import time
import pickle
from pyteomics import mzxml, auxiliary
import pandas as pd
import numpy as np
import re
import jcamp
import spc  # Module for reading, exploring and converting *.spc spectroscopic binary data in Python
from io import StringIO
import graphviz
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import export_graphviz
from numba import jit
from sklearn.cross_decomposition import PLSCanonical
from bisect import bisect_left
from contextlib import contextmanager

# constant
SMALL_NUM = 1e-10



def merge_csv(flist, **kwargs):
    """
        合并csv
    """
    return pd.concat([pd.read_csv(f, **kwargs) for f in flist], axis=1)


def psave(data=None, file_name=None):
    # 保存数据
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)


def pload(file_name=None):
    # 读取数据
    with open(file_name, 'rb') as f:
        return pickle.load(f)


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
    """
    正则匹配子字符串
    :param text:
    :param regexp:
    :return:
    """

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


def make_weights(x):
    """
    从x按列生成weight
    :param x: column vector或者按列排列的matrix
    :return: weights
    """
    x = np.array(x)
    if len(x.shape) < 2:
        x = x.reshape(-1, 1)
    weights = x/x.sum(axis=0)
    return weights


def plot_ConfusionMatrix(cm, sorted_unique_labels, normalize=True):
    """
    画混淆矩阵
    :param cm: confusionMatrix
    :param sorted_unique_labels: 各类的标签
    :param normalize: 是否normalize
    :return:
    """
    labels = sorted_unique_labels
    cm_norm = cm
    if normalize:
        try:
            cm_norm = cm/cm.sum(axis=1, keepdims=True)
        except: pass
    sns.heatmap(cm_norm, annot=cm, xticklabels=labels, yticklabels=labels)
    plt.xlabel('预测值')
    plt.ylabel('真实值')


def train_and_evaluate(X, labels, clf=None, params_grid=None,
                       verbose=True):
    """
    train and evaluate
    :param X:
    :param labels:
    :param clf:
    :param params_grid:
    :param verbose:
    :param kwargs:
    :return:
    """

    # default parameters
    if params_grid is None and clf is None:
        clf = KNeighborsClassifier()
        params_grid = {'n_neighbors': range(1, 5)}
    if params_grid is None and clf is not None:
        params_grid = {}

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.33)

    # train the model
    clf = GridSearchCV(clf, params_grid, n_jobs=-1, cv=3)
    clf = clf.fit(X_train, y_train)

    # evaluate the model
    if verbose:
        mean_accuracy = clf.score(X_test, y_test)
        print('test set mean accuracy is {}%'.format(mean_accuracy*100))
        cm = confusion_matrix(y_test, clf.predict(X_test))
        unique_labels = np.unique(y_test)
        plot_ConfusionMatrix(cm, unique_labels)
    return clf


def scatter(x, y, label):
    """
    scatter plot for group data
    :param x:
    :param y:
    :param label:
    :return:
    """
    for l in set(label):
        mask = label==l
        plt.scatter(x[mask], y[mask], label=l)
    plt.legend()


def cal_pct(mask):
    pct = sum(mask)/len(mask)
    return pct


def build_clf(X_train, y_train, cv=3):
    """

    :param X_train:
    :param y_train:
    :param cv
    :return:
    """
    clf = KNeighborsClassifier()
    params_grid = {'n_neighbors': range(1, 10)}
    clf = GridSearchCV(clf, params_grid, n_jobs=-1, cv=cv)
    clf = clf.fit(X_train, y_train)
    return clf


def read_hplc_csv(hplc_csv):
    with open(hplc_csv,'rb') as csv:
        hplc = pd.read_csv(csv,header=None,names=['retension_time','intensity']).set_index('retension_time')
        return hplc


def plot_tree(clf, feature_names):
    """
    plot decision tree, 画决策树
    plot_tree（dt, X.columns)
    :param clf:
    :param feature_names:
    :return:
    """
    dot_data = export_graphviz(clf, out_file=None,
                                feature_names=feature_names,
                                class_names=clf.classes_,
                                filled=True, rounded=True,
                                special_characters=True)
    return graphviz.Source(dot_data)


def plot_cm(X, y, estimator):
    """
    画混淆矩阵
    :param X:
    :param y:
    :param estimator:
    :return:
    """
    cm = confusion_matrix(y, estimator.predict(X))
    plot_ConfusionMatrix(cm, estimator.classes_)


def VIP(plsca):
    """
    variable importance in projection
    Camonical Powered PLS
    w: p*n_components 潜变量的权重
    SS: 1*m, the percentage of y explained by the m-th latent variable.
    """
    return _vip(plsca)


def _vip(plsca):
    t = plsca.x_scores_
    w = plsca.x_weights_
    q = plsca.y_loadings_
    p, h = w.shape
    vips = np.zeros((p,))

    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)

    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j]))**2 for j in range(h)])
        vips[i] = np.sqrt(p*(s.T @ weight)/total_s)

    return vips

def vcov(t, X, max=True):
    """
    计算v和X两两列协方差
    :param v:
    :param X:
    :return:
    """
    N = len(t)
    v = t.copy()
    if len(v.shape) == 1: v = np.reshape(v, (-1, 1))
    _X = X - np.mean(X, axis=0)
    if max: _X = _X/_X.max()
    _v = v - np.mean(v, axis=0)
    p1 = _v.T.dot(_X)/(N-1)
    return p1


def vcorr(t, X):
    """
    计算v和X两两列相关性
    相关性： cov/std,
    cov： 中心化后变量间的内积
    prod(std)： 中心化后变量自己的内积（平方和）的乘积
    url: https://zh.wikipedia.org/wiki/%E7%9A%AE%E5%B0%94%E9%80%8A%E7%A7%AF%E7%9F%A9%E7%9B%B8%E5%85%B3%E7%B3%BB%E6%95%B0
    :param v: m*n1
    :param X: m*n2
    :return: r: n1*n2
    """
    v = t.copy()
    if len(v.shape) == 1: v = np.reshape(v, (-1, 1))
    _X = X - np.mean(X, axis=0)
    _v = v - np.mean(v, axis=0)
    # square root of sum square
    sss_v = np.sqrt(np.sum(np.square(_v), axis=0))
    sss_X = np.sqrt(np.sum(np.square(_X), axis=0))
    assert len(sss_v.shape)==1
    din = sss_v[:, np.newaxis]*sss_X
    nume = _v.T.dot(_X)
    din[din == 0] = np.inf
    _corr = nume/din   # 保证din不能为0
    return _corr


def nan_ANOVA(X, y):
    """
    ANOVA 方差分析， f_oneway, 如果var为零则为np.nan
   """
    _x = X.copy()
    idx = _x.var() > 1e-5
    _x = _x.loc[:, idx]
    _F, _p = f_classif(_x, y)
    F = pd.Series(np.nan, index=idx.index)
    p = F.copy()
    F.loc[idx] = _F
    p.loc[idx] = _p
    return F, p


def test_VIP():
    # shape check
    np.random.seed(1)
    LVs = np.random.randn(10, 2)
    X = np.random.randn(10, 100)
    y = np.random.randn(10, 1)
    plsca = PLSCanonical(n_components=1).fit(X, y)
    vips = VIP(plsca)
    # todo numerical check


def test_vcorr():
    np.random.seed(1)
    v = np.linspace(1, 10, 10)
    X = np.linspace(2, 10, 10)
    r = vcorr(v, X[:, np.newaxis])
    assert(r == 1)


def test_ANOVA():
    y = np.random.randn(10) > 0
    X = np.random.randn(10, 100)
    X[:, 0] = 1
    F, p = nan_ANOVA(pd.DataFrame(X), 1*y)
    assert(np.isnan(F[0]))


def fold_change(a, b, type='log2'):
    """
    fold change
    """
    if type == 'log2':
        b = np.where(b > 0, b, np.ones_like(b) * np.nan)
        a = np.where(a > 0, a, np.ones_like(a) * np.nan)
        _fold_change = np.log2(b) - np.log2(a)
    return _fold_change


def interp1d(x, xp, yp):
    """
    1维线性插值
    :param x:
    :param xp:
    :param yp:
    :return:
    """
    # unique, sort
    _xp, index = np.unique(xp, return_index=True)
    _yp = [yp[i] for i in index]
    # inter
    _max, _min = max(xp), min(xp)

    #  sort xp
    # _xp, _yp= zip(*sorted(zip(xp, yp), key=lambda x: x[0]))
    slopes = [(_ypi1 - _ypi)/(_xpi1 - _xpi) for _xpi, _xpi1, _ypi, _ypi1 in zip(_xp, _xp[1:], _yp, _yp[1:])]
    y = [np.nan for xi in x]
    for ix in range(len(x)):
        _x = x[ix]
        if (_x >= _min) and (_x <= _max):
            ixp = bisect_left(xp, _x) - 1
            y[ix] = yp[ixp] + slopes[ixp]*(_x-xp[ixp])  # todo debug
    return y


def test_interp1():
    xp = [1, 2]
    yp = [3, 4]
    x = [1.5]
    y = interp1d(x, xp, yp)
    assert(np.isclose(y[0], 3.5))


def one_hot_encoder(df, cate_columns=None, na_as_category=True):
    """
    one hot encoding
    :param df:
    :param na_as_category:
    :param cate_columns
    :return:
    """
    object_cate = [column for column in df.columns if df[column].dtype == 'object']
    if cate_columns is not None:
        cate_columns = cate_columns + object_cate
    new_df = pd.get_dummies(df, columns=cate_columns, dummy_na=na_as_category)
    return new_df


@contextmanager
def timer(things):
    """
    计时器
    :param things:
    :return:
    """
    print(things + " starts" )
    start = time.time()
    yield start
    print(things + ' done in {}s'.format(round(time.time() - start, 2)))


if __name__ == '__main__':
    # from sklearn.cross_decomposition import PLSRegression, PLSCanonical
    X2 = pload('./data/testdata.p')
    # ms = metrics(x2, x2.index.values)

    def metrics(X, y):
        """
        计算Marker的metrics
        """
        _F, p_F = nan_ANOVA(X, y)
        plsca = PLSCanonical(n_components=1)
        plsca.fit(X2, X2.index.values)
        vip = VIP(plsca)
        mean_ = X.groupby(y).mean()
        fc_ = fold_change(mean_.iloc[0], mean_.iloc[1])
        r = vcorr(y, X.values)
        metrics = pd.DataFrame({'ANOVA_F': _F, 'ANOVA_p': p_F, 'VIP': vip, 'Fold_Change': fc_, 'Corr': r.flatten()})
        return metrics

    # ms = metrics(x2, x2.index.values)
    test_interp1()
