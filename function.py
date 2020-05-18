import torch
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

def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels

    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std

def _mat_sqrt(x):
    U, D, V = torch.svd(x)
    return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())


def coral(source, target):
    # assume both source and target are 3D array (C, H, W)
    # Note: flatten -> f

    source_f, source_f_mean, source_f_std = _calc_feat_flatten_mean_std(source)
    source_f_norm = (source_f - source_f_mean.expand_as(
        source_f)) / source_f_std.expand_as(source_f)
    source_f_cov_eye = \
        torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3)

    target_f, target_f_mean, target_f_std = _calc_feat_flatten_mean_std(target)
    target_f_norm = (target_f - target_f_mean.expand_as(
        target_f)) / target_f_std.expand_as(target_f)
    target_f_cov_eye = \
        torch.mm(target_f_norm, target_f_norm.t()) + torch.eye(3)

    source_f_norm_transfer = torch.mm(
        _mat_sqrt(target_f_cov_eye),
        torch.mm(torch.inverse(_mat_sqrt(source_f_cov_eye)),
                 source_f_norm)
    )

    source_f_transfer = source_f_norm_transfer * \
                        target_f_std.expand_as(source_f_norm) + \
                        target_f_mean.expand_as(source_f_norm)

    return source_f_transfer.view(source.size())
