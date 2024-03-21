import numpy as np

data = np.array([[0,1,1],
                [0,0,0],
                [1,0,0],
                [1,1,0],
                [1,0,1],
                [0,0,1],
                [0,1,0],
                [1,1,1]])
x1 = np.transpose(data)[0]
x2 = np.transpose(data)[1]
x3 = np.transpose(data)[2]
trues = np.array([0,0,0,0,0,0,0,1])
N = len(trues)
w1 = 0
w2 = 0
w3 = 0
w0 = 0
lr = .5

def logsig(a):
    return 1/(1+np.exp(-a))

epochs = 20000
pred = []
for i in range(epochs):
    pred = logsig(w1*x1 + w2*x2 + w3*x3+w0)

    sum_w1 = np.sum(2*(trues-pred)*-pred*(1-pred)*x1)
    nabla_w1 = 1/N*sum_w1
    w1 = w1 - lr*nabla_w1
    nabla_w2 = 1/N*np.sum(2*(trues-pred)*-pred*(1-pred)*x2)
    w2 = w2 - lr*nabla_w2
    sum_w3 = np.sum(2*(trues-pred)*-pred*(1-pred)*x3)
    nabla_w3 = 1/N*sum_w3
    w3 = w3 - lr *nabla_w3
    nabla_w0 = 1/N*np.sum(2*(trues-pred)*-pred*(1-pred))
    w0 = w0 - lr *nabla_w0

    if i % 2000 == 0:
        print(1/N*np.sum((trues-pred)**2))
    
print(pred)
print(trues[7]-pred[7])