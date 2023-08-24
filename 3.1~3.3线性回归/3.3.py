import torch
import random
from torch.utils import data
import torch.nn as nn

alpha=0.1
def synthetic_data(w, b, num_examples):
    X=torch.normal(0,0.01,(num_examples,len(w)))
    Y=torch.mv(X,w)+b
    c=torch.normal(0,0.01,(Y.shape))
    Y+=c
    return X,Y.reshape((-1,1))

def load_data(data_arrays,batch_size,is_train=True):
    dataset=data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset,batch_size,is_train)

def huber(prediction,label,alpha):
    siz=len(prediction)
    total_error=0
    for i in range(siz):
        error=0
        y_hat=prediction[i]
        y=label[i]
        if(abs(y_hat-y)>alpha):
            error= abs(y_hat-y)-0.5*alpha
        else:
            error=(y_hat-y)**2/2/alpha
        total_error+=error
    return total_error/siz

true_w=torch.tensor([2,-3.4])
true_b=4.2
features,labels=synthetic_data(true_w,true_b,1000)

batch_size=10
data_iter=load_data((features,labels),batch_size)

print(next(iter(data_iter)))


net=nn.Sequential(nn.Linear(2,1))
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)

loss=nn.HuberLoss()
trainer=torch.optim.SGD(net.parameters(),lr=0.03)

num_empochs=3
for epoch in range(num_empochs):
    for X,Y in data_iter:
        Y_hat=net(X)
        #l=huber(Y_hat,Y,0.1)
        l=loss(Y_hat,Y)
        trainer.zero_grad() #一定要加，不然梯度累计
        l.backward()
        trainer.step()
    print(f"epoch:{epoch+1} loss:{loss(net(features),labels)}")