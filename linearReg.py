import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as pt

X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise = 20, random_state=1)
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0],1)

n_samples,n_features = X.shape

input_size = n_features
output_size = n_features
model = nn.Linear(input_size,output_size)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters() ,lr = 0.01)

#traing
num_epcohs = 100

for epoch in range(num_epcohs):
    # forward
    y_pred = model(X)
    loss = criterion(y_pred, y)
     # backward
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    if (epoch+1) % 10 ==0:
        print(f'epoch: {epoch+1}, loss ={loss.item():.4f}')

pred = model(X).detach().numpy() # convert to numpy so detach
pt.plot(X_numpy,y_numpy,'ro')
pt.plot(X_numpy,pred,'b')
pt.show()