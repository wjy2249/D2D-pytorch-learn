import torch
import torch.nn as nn
import math
import numpy as np
from matplotlib import pyplot as plt
import os
from mpl_toolkits.mplot3d import axes3d
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

x=torch.arange(0,10,0.1,requires_grad=True)
y=torch.sin(x)
y.backward(torch.ones_like(y))
z=x.grad
print(z)

xn=x.detach().numpy()
yn=y.detach().numpy()
zn=z.numpy()
# print(xn)
# print(yn)

plt.xlabel("x")
plt.ylabel("y")


plt.plot(xn,yn,label="sin(x)")
plt.plot(xn,zn,label="sin'(x)")
plt.legend()

plt.show()