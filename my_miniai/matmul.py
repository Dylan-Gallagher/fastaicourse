def chunks(x, sz):
    for i in range(0, len(x), sz):
        yield x[i:i + sz]


class Matrix:
    def __init__(self, xs):
        self.xs = xs

    def __getitem__(self, idxs):
        return self.xs[idxs[0]][idxs[1]]