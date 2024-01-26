import math, torch, matplotlib.pyplot as plt
from operator import attrgetter
from torch import optim
import torch.nn.functional as F
import matplotlib as mpl
import torchvision.transforms.functional as TF
from torch import nn, tensor
from datasets import load_dataset, load_dataset_builder
from miniai.datasets import *
import logging

# %%
torch.set_printoptions(precision=2, linewidth=140, sci_mode=False)
torch.manual_seed(1)
mpl.rcParams['image.cmap'] = 'gray'
# %%
logging.disable(logging.WARNING)
# %% md
## Learner
# %%
x, y = 'image', 'label'
name = "fashion_mnist"
dsd = load_dataset(name)


# %%
@inplace
def transformi(b): b[x] = [torch.flatten(TF.to_tensor(o)) for o in b[x]]


# %%
bs = 1024
tds = dsd.with_transform(transformi)
# %%
dls = DataLoaders.from_dd(tds, bs, num_workers=4)
dt = dls.train
xb, yb = next(iter(dt))


class CancelFitException(Exception):
    pass


class CancelBatchException(Exception):
    pass


class CancelEpochException(Exception):
    pass


class Callback():
    order = 0


def run_cbs(cbs, method_nm, learn=None):
    for cb in sorted(cbs, key=attrgetter('order')):
        method = getattr(cb, method_nm, None)
        if method is not None: method(learn)


class CompletionCB(Callback):
    def before_fit(self, learn):
        self.count = 0

    def after_batch(self, learn):
        self.count += 1

    def after_fit(self, learn):
        print(f'Completed {self.count} batches')


cbs = [CompletionCB()]
run_cbs(cbs, 'before_fit')
run_cbs(cbs, 'after_batch')
run_cbs(cbs, 'after_fit')


class Learner():
    def __init__(self, model, dls, loss_func, lr, cbs, opt_func=optim.SGD):
        self.model = model
        self.dls = dls
        self.loss_func = loss_func
        self.lr = lr
        self.cbs = cbs
        self.opt_func = opt_func

    def one_batch(self):
        self.preds = self.model(self.batch[0])
        self.loss = self.loss_func(self.preds, self.batch[1])
        if self.model.training:
            self.loss.backward()
            self.opt.step()
            self.opt.zero_grad()

    def one_epoch(self, train):
        self.model.train(train)
        self.dl = self.dls.train if train else self.dls.valid
        try:
            self.callback('before_epoch')
            for self.iter, self.batch in enumerate(self.dl):
                try:
                    self.callback('before_batch')
                    self.one_batch()
                    self.callback('after_batch')
                except CancelBatchException: pass
            self.callback('after_epoch')
        except CancelEpochException: pass

    def fit(self, n_epochs):
        self.n_epochs = n_epochs
        self.epochs = range(n_epochs)
        self.opt = self.opt_func(self.model.parameters(), self.lr)
        try:
            self.callback('before_fit')
            for self.epoch in self.epochs:
                self.one_epoch(True)
                self.one_epoch(False)
            self.callback('after_fit')
        except CancelFitException: pass

    def callback(self, method_nm):
        run_cbs(self.cbs, method_nm, self)


m = 28 * 28
nh = 50


def get_model():
    return nn.Sequential(nn.Linear(m,nh), nn.ReLU(), nn.Linear(nh,10))


model = get_model()
learn = Learner(model, dls, F.cross_entropy, lr=0.2, cbs=[CompletionCB()])
learn.fit(1)
