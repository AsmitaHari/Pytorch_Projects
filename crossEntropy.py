import torch
import torch.nn as nn
import numpy as np

def ce(actual, pred):
    return -np.sum(actual * np.log(pred))


y = np.array([1,0,0])

y_pred_good = np.array([0.7,0.2,0.1])
y_pred_bad = np.array([0.1,0.3,0.6])

print(f'looss: {ce(y,y_pred_good)}')
print(f'looss: {ce(y,y_pred_bad)}')


loss = nn.CrossEntropyLoss()

y = torch.tensor([0])

#nsamples * nclasses = 1* 3

y_pred_good = torch.tensor([[2.0,1.0,0.1]])
y_pred_bad = torch.tensor([[0.5,2.0,0.3]])

l1 = loss(y_pred_good,y)
l2 = loss(y_pred_bad,y)

_, predic1 = torch.max(y_pred_good,1)
_, predic2 = torch.max(y_pred_bad,1)

print(l1.item())
print(l2.item())