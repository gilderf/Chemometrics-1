import numpy as np
import pycrfsuite
import sklearn_crfsuite

# ---峰识别pick peaking---
# target map
MAP = {0: 'baseline',
       1: 'up',
       2: 'down'}

def noise_intensity(x):
    """
    获取基线强度: 前10和后10个点的median
    :return:
    """
    nf = np.median(np.hstack([x[:10], x[-10:]]))
    return nf


def crf_peaking(model, intensity, Nf=None):
    """
    峰识别
    :return:
    """
    if Nf is None: Nf = noise_intensity(intensity)
    fs = intensity2features(intensity, Nf)
    tagger = pycrfsuite.Tagger()
    tagger.open(model)
    ypred = tagger.tag(fs)
    return ypred


def intensity2features(intensity, Nf=None):
    """
    特征生成

    特征 - [{特征名称：特征值}]
    :param intensity:
    :param Nf:
    :return:
    """
    if Nf is None: Nf = noise_intensity(intensity)
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


def train_crf_peaking(intensity, target, model_name=None, Nf=None, reverse=True, style='sklearn'):
    """
    训练峰识别模型
    :param intensity:
    :param target:
    :return:
    """

    # 生成数据，prepare features
    if Nf is None: Nf = noise_intensity(intensity)
    targets = [MAP[v] for v in target]
    features = intensity2features(intensity, Nf=Nf)
    if reverse:
        features = [features, features[-1::-1]]
        targets = [targets, targets[-1::-1]]

    # CRFs模型
    trainer = pycrfsuite.Trainer()
    trainer.set_params({
        'c1': 1.0,  # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 100,  # stop earlier
        # include transitions that are possible, but not observed
        'feature.possible_transitions': False
    })

    # 喂数据
    for i in range(len(features)):
        trainer.append(features[i], targets[i])

    # 训练模型
    if model_name is not None:
        trainer.train(model_name)
    else:
        trainer.train()
    return trainer


if __name__ == '__main__':
    # 测试放在./tests/test_chromatogram.py
    crf = sklearn_crfsuite.CRF(c1=1, c2=1e-3, max_iterations=100, all_possible_transitions=False)
    crf.fit(features, targets)
