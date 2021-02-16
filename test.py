from __future__ import print_function
import torch
x = torch.rand(2,2)
y = torch.rand(2,2)

print(x)
print(y)

z = x+y

print(z)
print(y.add_(x))

print(z[1:])
