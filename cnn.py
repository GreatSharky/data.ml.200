from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
import torch.utils.data as torchdata
from torch import nn
import torch
import os

BATCH = 32

device = torch.device('cuda')

root = "GTSRB_subset_2"
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), root)

dataset = ImageFolder(root=path,transform=transforms.ToTensor())

data = torchdata.random_split(dataset, [.8,.2], generator=torch.Generator().manual_seed(1))
train_data = data[0]
test_data = data[1]

train_data = torchdata.DataLoader(dataset=train_data, batch_size=BATCH, shuffle=True)
test_data = torchdata.DataLoader(dataset=test_data,batch_size=BATCH, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.full_stack = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=90, out_features=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        oput = self.full_stack(x)
        return oput
    

model = CNN().to(device)
print(model)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1)

def train(data, model, loss_fn, optimizer):
    model.train()
    for batch, (X,y) in enumerate(data):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred,y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def test(data, model, loss_fn):
    num_batches = len(data)
    size = len(data.dataset)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for X, y in data:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    acc = correct/size
    print(f"Acc: {acc} Avg loss: {test_loss:>8f}")

epochs = 20
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_data, model, loss_func, optimizer)
    test(test_data, model, loss_func)
print("Done!")