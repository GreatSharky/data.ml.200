import numpy as np

def logsig(x):
    return 1/(1+np.exp(-x))

class Perceptron:
    def __init__(self, weights: list, weight0: float, lr, inputs, y_true) -> None:
        self.weights = weights
        self.bias = weight0
        self.inputs = inputs
        self.trues = y_true
        self.pred = []
        self.lr = lr
    
    def forward_pass(self):
        a = self.bias
        self.pred = []
        for i in self.inputs:
            for j , w in enumerate(self.weights):
                a += self.weights[j] * i[j]
            self.pred.append(logsig(a))
        return self.pred

    def update(self):
        for j, w in enumerate(self.weights):
            sum = 0
            for i, point in enumerate(self.inputs):
                sum += 2*(self.trues[i]-self.pred[i])*-self.pred[i]*(1-self.pred[i])*point[j]
            
            w_new = w - self.lr/len(self.inputs)*sum
            self.weights[j] = w_new
        for i, point in enumerate(self.inputs):
            sum += 2*(self.trues[i]-self.pred[i])*-self.pred[i]*(1-self.pred[i])
        self.bias = self.bias - self.lr/len(self.inputs)*sum
        return sum
    
def main():
    w = [0,0,0]
    data = np.array([[0,1,1],
                  [0,0,0],
                  [1,0,0],
                  [1,1,0],
                  [1,0,1],
                  [0,0,1],
                  [0,1,0],
                  [1,1,1]])#1,0,0,0,1,0,0,0

    lr = .5
    epochs = 2000
    trues_a = np.array([0,0,0,0,0,0,0,1])
    a = Perceptron(w, 0, lr, data, trues_a)
    N = len(trues_a)
    
    trues_b = np.array([0,1,1,1,1,1,1,1])
    b = Perceptron(w,0,lr,data,trues_b)

    trues_c = np.array([1,0,0,0,1,0,0,0])
    c = Perceptron(w,0,lr,data,trues_c)
    for j in range(epochs):
        preds_a = a.forward_pass()
        a.update()
        preds_b = b.forward_pass()
        b.update()
        preds_c = c.forward_pass()
        c.update()

    print(f"x1 AND x2 AND x3: (After {epochs} epochs)")
    print("\tGround truth:", trues_a)
    print("\tPrediction:", preds_a)
    print("\tMSE:", 1/N*np.sum((trues_a-preds_a)**2))

    print(f"\nx1 OR x2 OR x3: (After {epochs} epochs)")
    print("\tGround truth:", trues_b)
    print("\tPrediction:", preds_b)
    print("\tMSE:", 1/N*np.sum((trues_b-preds_b)**2))

    print(f"\n((x1 AND !x2) OR (!X1 AND x2))AND x3: (After {epochs} epochs)")
    print("\tGround truth:", trues_c)
    print("\tPrediction:", preds_c)
    print("\tMSE:", 1/N*np.sum((trues_c-preds_c)**2))




main()