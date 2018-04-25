import numpy as np
import matplotlib.pyplot as plt
import math
from Chemometrics.basic import plot1
import pycrfsuite
from sklearn.metrics import classification_report

# 常量
PI = math.pi


def normal(x, mu, sigma):
    """
    正态分布函数
    :param x:
    :param mu:
    :param sigma:
    :return:
    """
    y = (1/np.sqrt(2*PI*sigma**2))*np.exp(-(x-mu)**2/(2*sigma**2))
    return y


def intensity2features(intensity, Nf):
    """
    特征生成

    :param intensity:
    :param Nf:
    :return:
    """
    f1 = intensity > 3 * Nf
    f2 = np.full(intensity.shape, False, dtype=bool)
    f2[:-2] = intensity[:-2] < intensity[2:]  # vi < vi+2
    f3 = np.full(intensity.shape, False, dtype=bool)  # vi-2 < vi
    f3[2:] = intensity[:-2] < intensity[2:]
    fs = [{'>3Nf': {str(f1[i]): f1[i]},
           '<i+2': {str(f2[i]): f2[i]},
           '>i-2': {str(f3[i]): f3[i]},
           'intensity': intensity[i]
           }
          for i in range(len(intensity))]
    return fs


def gen_sample(timerange, cs, mus, sigmas, b, bstd, peaks_start_intensity =2.5):
    """
    生成样本
    """
    intensitys = [cs[i] * normal(timerange, mus[i], sigmas[i])for i in range(len(cs))]
    thresholds = [cs[i]*normal(mus[i]-peaks_start_intensity*sigmas[i], mus[i], sigmas[i]) for i in range(len(cs))]
    intensity = np.vstack(intensitys).sum(axis=0)
    target = get_target(intensity, thresholds, b)
    baseline = bstd*np.random.randn(len(intensity)) + b
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
    target[(intensity>t) & (dif > 0)] = 1
    target[(intensity > t) & (dif < 0)] = 2
    return target


if __name__ == '__main__':

    # peak
    b = 1
    time = np.linspace(0, 20, 100)
    intensity, target = gen_sample(time, [10, 5], [10, 14], [2, 1], b=1, bstd=.1)

    # baseline
    # intensity_b = .1*np.random.randn(len(intensity)) + b
    # plt.plot(time, intensity)
    # plt.plot(time, intensity_b)
    # plt.plot(time, intensity + intensity_b)
    # plt.show()

    # target
    _map = {0: 'baseline',
            1: 'up',
            2: 'down'}
    # target = np.zeros_like(intensity)
    # a = [5.4, 9.9, 11.95, 13.95, 16.6]
    # target[(time >= 5.4) & (time < 9.9)] = 1
    # target[(time >= 9.9) & (time < 11.95)] = 2
    # target[(time >= 11.95) & (time < 13.95)] = 1
    # target[(time >= 13.95) & (time < 16.6)] = 2
    # target[time >= 16.6] = 0



    #  ----crf
    # prepare features
    target_str = [_map[v] for v in target]
    Nf = b
    fs = intensity2features(intensity, Nf=b)

    # train
    trainer = pycrfsuite.Trainer(verbose=True)
    features = [fs, fs[-1::-1]]
    targets = [target_str, target_str[-1::-1]]
    for i in range(len(features)):
        trainer.append(features[i], targets[i])
    trainer.set_params({
        'c1': 1.0,  # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 100,  # stop earlier
        # include transitions that are possible, but not observed
        'feature.possible_transitions': False
    })
    trainer.train('peak_idetify')

    # predict
    tagger = pycrfsuite.Tagger()
    tagger.open('peak_idetify')
    intensity_, target_ = gen_sample(time, [11, 6], [10, 17], [2, 1], b=1.2, bstd=.1)
    fs_ = intensity2features(intensity_, Nf=b)
    target_str_ = [_map[v] for v in target_]
    ypred = tagger.tag(fs_)
    print(np.unique(ypred))
    print(classification_report(target_str_, ypred))
    # list(zip(ypred, target_str_))
