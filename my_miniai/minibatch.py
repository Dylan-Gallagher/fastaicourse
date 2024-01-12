import math
import torch


def log_sum_exp(x):
    a = x.max(-1)[0]
    return a + (x-a[:,None]).exp().sum(-1).log()


def log_softmax(x):
    return x - log_sum_exp(x)


def accuracy(out, yb):
    return (out.argmax(dim=1) == yb).float().mean()


def report(loss, preds, yb):
    print(f'{loss:.2f}, {accuracy(preds, yb):.2f}')







