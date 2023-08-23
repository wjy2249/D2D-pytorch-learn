import torch
import torch.nn as nn
import math
import numpy as np
from matplotlib import pyplot as plt
import os
from mpl_toolkits.mplot3d import axes3d

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def fun(x):
    return x**3-1/x

def numerical_lim(f,x,h):#求导数
    return (f(x+h)-f(x))/h

def targent_line(f,x):#求切线
    k=numerical_lim(f,x,0.01)
    b=f(x)-k*x
    return lambda t:k*t+b
def gradeint(f,x):
    h=0.01
    grad=np.zeros_like(x)
    for idx,element in np.ndenumerate(x):
        temp=x.copy()#直接赋值会指向同一个内存，到时候值会一起变
        temp[idx]+=h
        grad[idx]=(f(temp)-f(x))/h
    return grad

def fun2(x):
    return 3*x[0]+5*math.exp(x[1])

def fun3(x):
    return np.linalg.norm(x)

def fun4(x,y):
    return 3*x+np.exp(y)

print(1)
x=np.arange(0.1,3,0.1)
y=fun(x)

plt.xlabel("x")
plt.ylabel("y")

tf=targent_line(fun,1)
y2=tf(x)

plt.plot(x,y,label='f(x)')
plt.plot(x,y2,label='targentline_x=1')

plt.legend()#用于添加图例
plt.show()#Q1 end
plt.clf()

x1=np.arange(0,3,0.1)
y1=np.arange(0,3,0.1)
X,Y=np.meshgrid(x1,y1)

print(gradeint(fun2,np.array([2.0,3.0])))#Q2 end

print(gradeint(fun3,np.array([2.0,3.0])))#Q2 end

fig=plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

X,Y=np.meshgrid(x1,y1)
Z=3*X+5*np.exp(Y)

z=fun4(x1,y1)

ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')
ax1.set_zlabel('z axis')

ax1.plot(x1,y1,z) #绘制三维曲线
plt.show()
plt.clf()

fig=plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

ax1.plot_surface(X,Y,Z,cmap='viridis')#绘制3D曲面
plt.show()

