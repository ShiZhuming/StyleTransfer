import numpy as np
from torch.utils import data



def RecurrentSample(n):
    '''循环采样'''
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class RecurrentSampler(data.sampler.Sampler):
    '''循环采样器'''
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(RecurrentSample(self.num_samples))

    def __len__(self):
        return 2 ** 31


def get_mean_std(feature, epsilon=1e-5):
    '''计算特征图的均值与标准差'''
    # epsilon用来防止出现除零运算
    N, C = feature.size()[:2]
    feature_var = feature.view(N, C, -1).var(dim=2) + epsilon#计算方差
    feature_std = feature_var.sqrt().view(N, C, 1,1)#计算偏移标准差
    feature_mean = feature.view(N, C, -1).mean(dim=2).view(N, C, 1,1)

    return feature_mean, feature_std

def AdaIN(content_features,style_features,epsilon=1e-5):
    '''归一化层，用于调整图片样式的核心层'''
    assert (content_features.size()[:2] == style_features.size()[:2])
    content_mean,content_std=get_mean_std(content_features)
    style_mean,style_std=get_mean_std(style_features)

    size = content_features.size()
    normalized_features=(content_features-content_mean.expand(size))/content_std.expand(size)
    normalized_features=(normalized_features*style_std.expand(size))+style_mean.expand(size)

    return normalized_features

    