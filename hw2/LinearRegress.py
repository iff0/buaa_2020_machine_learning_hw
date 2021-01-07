import numpy as np


class LinearRegress():
    def __init__(self, x: np.ndarray, y: np.ndarray, LR=0.005, verbose=5):
        self.x = x
        self.y = y
        self.w = np.absolute(np.random.randn(x.shape[1], 1)) * 4000
        self.b = 0.
        self.LR = LR
        self.verbose = verbose
        self.r = []

    def predict(self, x):
        return x.dot(self.w) + self.b

    def train(self, epoches):
        cnt = 0
        vnt = 0

        while cnt < epoches:
            dy = self.predict(self.x) - self.y
            #print(self.x)
            #print(self.predict(self.x))
            #print(np.square(dy))
            loss = np.mean(np.square(dy))
            # self.r.append((dy, loss))
            self.w -= self.LR * (self.x.T.dot(dy)) / self.x.shape[0]
            self.b -= self.LR * np.mean(dy)
            cnt += 1
            if cnt / epoches > vnt / self.verbose or cnt == epoches - 1:
                print('process: %f, loss: %f' % (vnt / self.verbose, loss))
                vnt += 1



if __name__ == '__main__':
    # debug

    x = np.array([[1, 2], [2, 3], [3, 4], [4, 5], ])
    y = np.array([2, 4, 6, 8]).reshape(4, 1)
    lr = LinearRegress(x, y)
    lr.train(10000)
    print("w: ", lr.w , "b: %f" % lr.b)
    pass
