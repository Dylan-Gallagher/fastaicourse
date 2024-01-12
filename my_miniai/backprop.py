import torch
import math


class Module:
    def __call__(self, *args):
        self.args = args
        self.out = self.forward(*args)
        return self.out

    def forward(self):
        raise Exception("Not implemented")

    def backward(self):
        self.bwd(self.out, *self.args)

    def bwd(self):
        raise Exception("Not implemented")


class Relu(Module):
    def forward(self, inp):
        return inp.clamp_min(0.)

    def bwd(self, out, inp):
        inp.g = (inp > 0).float() * out.g


class Lin(Module):
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def forward(self, inp):
        return inp @ self.w + self.b

    def bwd(self, out, inp):
        inp.g = self.out.g @ self.w.t()
        self.w.g = inp.t() @ self.out.g
        self.b.g = self.out.g.sum(0)


class Mse(Module):
    def forward(self, inp, targ):
        return (inp.squeeze() - targ).pow(2).mean()

    def bwd(self, out, inp, targ):
        inp.g = (2. * (inp.squeeze() - targ).unsqueeze(-1)
                 / targ.shape[0])


class Model:
    def __init__(self, w1, b1, w2, b2):
        self.layers = [Lin(w1, b1),
                       Relu(),
                       Lin(w2, b2)]
        self.loss = Mse()

    def __call__(self, x, targ):
        for l in self.layers:
            x = l(x)
        return self.loss(x, targ)

    def backward(self):
        self.loss.backward()
        for l in reversed(self.layers):
            l.backward()
