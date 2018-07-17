import numpy as np


def COW(sample, reference, segment_length=4, slack=1):
    """
    correlation optimized warping
    :param sample:
    :param reference:
    :param segment_length:
    :param slack:
    :return:
    """
    L = len(reference)
    num_seg = int(L / segment_length)
    border_ref = range(0, L, segment_length)[:num_seg][-1::-1]  # last towards the first
    last_length = L - border_ref[0]



    # 一次搜索
    border = None  # placeholder
    borders = range(border - slack, border + slack+1)
    start = 0 # placeholder
    ref_ = reference[i]
    length_target = len(ref_)
    r = []
    for b in borders:
        sample_ = sample[b:start+1]
        sample_ = linear_inter(sample_, length_target)
        r += np.corrcoef(sample_, ref_)
    # border状态？


def linear_inter(y, length_target):
    """
    linear interpolate
    :param y: knots
    :param length_target: 输出长度
    :return:
    """
    length_y = len(y)
    if length_y == length_target:
        return y
    x = np.linspace(0, 1, length_y)
    x_ = np.linspace(0, 1, length_target)
    y_ = np.interp(x_, x, y)
    return y_

