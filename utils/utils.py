import torch
import numpy as np
import random
from functools import partial

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def setup_seed(rs):
    """
    set random seed for reproducing experiments
    :param rs: random seed
    :return: None
    """
    torch.manual_seed(rs)
    torch.cuda.manual_seed_all(rs)
    np.random.seed(rs)
    random.seed(rs)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def copy(target, source):
    for name in target:
        target[name].data = source[name].data.clone()


def add(target, source):
    # for name in target:
    #     target[name].data += source[name].data.clone()
    # partially update, only update some components
    for name in source:
        target[name].data += source[name].data.clone()


def subtract(target, source):
    for name in target:
        target[name].data -= source[name].data.clone()


def subtract_(target, minuend, subtrahend):
    for name in target:
        target[name].data = minuend[name].data.clone() - subtrahend[name].data.clone()


def average(target, sources):
    for name in target:
        target[name].data = torch.mean(torch.stack([source[name].data for source in sources]), dim=0).clone()


def weighted_average(target, sources, weights):
    for name in target:
        summ = torch.sum(weights)
        n = len(sources)
        modify = [weight / summ * n for weight in weights]
        target[name].data = torch.mean(torch.stack([m * source[name].data for source, m in zip(sources, modify)]),
                                       dim=0).clone()


def majority_vote(target, sources, lr):
    for name in target:
        threshs = torch.stack([torch.max(source[name].data) for source in sources])
        mask = torch.stack([source[name].data.sign() for source in sources]).sum(dim=0).sign()
        target[name].data = (lr * mask).clone()


#############################################
# Compressor
#############################################
def topk(T, params):
    threshold = params['threshold']
    T[T.abs() < threshold] = 0.0
    return T


def compress(target, source, hp):
    """
    compress_fun : a function f : tensor (shape) -> tensor (shape)
    """
    compressor = hp['compressor']
    params = {}
    if compressor == 'topk':
        # topk, find threshold corresponding to top K
        percentage = hp['percentage']
        threshold = findK(source, percentage)
        params.update({'threshold': threshold})
    compress_fun = compression_function(name=compressor, params=params)
    for name in target:
        target[name].data = compress_fun(source[name].data.clone())


def findK(source, percentage):
    out = torch.Tensor().to(device=device)
    for name, value in source.items():
        out = torch.cat((out, value.flatten()), dim=0)

    # absolute value
    out = out.abs()
    # print(out.shape)

    k = int(percentage * out.shape[0])
    values, indices = torch.topk(out, k=k, sorted=True)
    # topk element -> threshold
    threshold = values[-1]
    return threshold


def compression_function(name, params=None):
    """
    Returns a function that maps a tensor to a tensor of the same shape
    """
    return partial(globals()[name], params=params)


def getBits(T, compressor):
    """
    return the number of bits that are required to communicate the Tensor T, which was compressed with compressor
    :param T:
    :param compressor:
    :return:
    """
    B_pos = None
    B_val = {
        'none': 32,
        'topk': 32
    }[compressor]

    k = None
    # dense compressor
    if compressor in ['none']:
        k = T.numel()
        B_pos = 0

    # sparse compressor non-optimal encoding
    elif compressor in ['topk']:
        k = torch.sum(T != 0.0).item()
        B_pos = 16

    b_total = (B_pos + B_val) * k
    return b_total


def getUpdateSize(dW, compressor):
    """
    return the number of bits that are required to communicate the entire network dW, which was compressed with compression method
    """
    updateSize = sum(getBits(T, compressor=compressor) for T in dW.values())
    return updateSize
