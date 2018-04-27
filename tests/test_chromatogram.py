from Chemometrics.chromatogram import *
from math import pi as PI
import sklearn_crfsuite


def gen_sample(timerange, cs, mus, sigmas, b, bstd, peaks_start_intensity=2.5):
    """
    生成样本
    """
    intensitys = [cs[i] * normal(timerange, mus[i], sigmas[i]) for i in range(len(cs))]
    thresholds = [cs[i] * normal(mus[i] - peaks_start_intensity * sigmas[i], mus[i], sigmas[i]) for i in
                  range(len(cs))]
    intensity = np.vstack(intensitys).sum(axis=0)
    target = get_target(intensity, thresholds, b)
    baseline = bstd * np.random.randn(len(intensity)) + b
    intensity = intensity + baseline
    return intensity, target


def get_target(intensity, thresholds, b):
    """
    get_target
    :param intensity:
    :param thresholds:
    :param b:
    :return:
    """
    thresholds.append(b)
    t = max(thresholds)
    target = np.zeros_like(intensity)
    dif = np.zeros_like(intensity)
    dif[1:] = np.diff(intensity)
    target[(intensity > t) & (dif > 0)] = 1
    target[(intensity > t) & (dif < 0)] = 2
    return target


def normal(x, mu, sigma):
    """
    正态分布函数
    :param x:
    :param mu:
    :param sigma:
    :return:
    """
    y = (1 / np.sqrt(2 * PI * sigma ** 2)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    return y


# 生成数据
b = 1
time = np.linspace(0, 20, 100)
intensity, target = gen_sample(time, [10, 5], [10, 14], [2, 1], b=b, bstd=.1)
model_name = 'test_crf_peaking'
targets = [MAP[v] for v in target]
features = intensity2features(intensity, Nf=b)


def test_peaking():
    """
    测试peaking
    :return:
    """
    train_crf_peaking(intensity, target, model_name=model_name)
    crf_peaking(model_name, intensity)


def tskCRF():
    """
    test sklearn CRFsuit
    :return:
    """
    crf = sklearn_crfsuite.CRF(c1=1, c2=1e-3, max_iterations=100, all_possible_transitions=False)
    crf.fit(features, targets)
    crf.predict(features)