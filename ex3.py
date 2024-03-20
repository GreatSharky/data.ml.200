import numpy as np

def logsig(x):
    return 1/(1+np.exp(-x))

class Perceptron:
    def __init__(self, weights: list, weight0: float, lr) -> None:
        self.weights = weights
        self.bias = weight0
        self.lr = lr
    
    def forward_pass(self, inputs):
        a = self.bias
        self.inputs = inputs
        y_pred = []
        for i in self.inputs:
            for j , w in enumerate(self.weights):
                a += self.weights[j] * i[j]
            y_pred.append(logsig(a))
        return y_pred

    def update(self, y_pred, y_true):
        for j, w in enumerate(self.weights):
            sum = 0
            for i, y in enumerate(y_pred):
                sum += (y_true[i] - y_pred[i])*(y_true[i]-y_pred[i]*(1-y_pred[i])*4)
            w_new = self.weights[j] - self.lr*2/len(y_pred)*sum
            self.weights[j] = w_new
        for i, y in enumerate(y_pred):
            sum += (y_true[i] - y_pred[i])*(y_true[i]-y_pred[i]*(1-y_pred[i]))
        self.bias = self.bias - self.lr*2/len(y_pred)
        return self.bias, self.weights

def main():
    w = [5,5,5]
    data = np.array([[0,1,1],
                  [0,0,0],
                  [1,0,0],
                  [1,1,0],
                  [1,0,1],
                  [0,0,1],
                  [0,1,0],
                  [1,1,1]])

    lr = .2
    a = Perceptron(w,-5,lr)
    epoch = 2010
    y_t = np.array([0,0,0,0,0,0,0,1])
    target = .4
    for j in range(epoch):
        y_p = a.forward_pass(data)
        a.update(y_p, y_t)
        if j % 500 == 0:
            print(y_p)

main()