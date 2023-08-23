import torch
import torch.nn as nn
import math
import numpy as np

x=torch.arange(4.0)
x.requires_grad_(True)
print(x.grad)

y=x*x
print(y)
y.sum().backward(retain_graph=True)
print(x.grad)
x.grad.zero_()
print(x.grad)
y.backward(torch.ones_like(y),retain_graph=True)
print(x.grad)

x.grad.zero_()
print(x.grad)
z=torch.zeros(3,4)
z[0]=x*x
z[1]=2*x;
z[2]=x**3
z.backward(torch.ones_like(z))
print(x.grad)

def f(a):
    x=a
    for i in range(0,5):
        x=x*2
    return x

a=torch.ones(1,requires_grad=True)
b=f(a)
b.backward()
print(a.grad)#控制流看梯度

