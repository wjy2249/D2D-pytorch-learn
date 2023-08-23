import torch
import torch.nn as nn

device="cuda"

A=torch.ones(3,4)
x=torch.tensor([1.,2.,3.,4.],dtype=torch.float32)
# B=A
# B+=1
B=A.clone()
B+=1
print(A,B)

C=A*x
D=torch.mv(A,x)
print(C,D)

print(A.T.T==A)
print(A.T+B.T==(A+B).T)

X=torch.arange(24,dtype=torch.float32).reshape(2,3,4)
print(len(X))

print(X.sum(axis=1),X.sum(axis=1,keepdim=True))
print(X/X.sum(axis=1,keepdim=True))

print(X.sum(0),"\n",X.sum(1),"\n",X.sum(2))

print(torch.linalg.norm(X,ord=None))