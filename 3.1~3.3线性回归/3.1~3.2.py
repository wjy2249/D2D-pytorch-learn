import torch
import random

def synthetic_data(w, b, num_examples):
    X=torch.normal(0,0.01,(num_examples,len(w)))
    Y=torch.mv(X,w)+b
    c=torch.normal(0,0.01,(Y.shape))
    Y+=c
    return X,Y.reshape((-1,1))

def data_iterator(batch_size,features,labels):
    num_examples=len(labels)
    indices=list(range(num_examples))
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        batch_indices=torch.tensor(indices[i:min(i+batch_size,num_examples)])
        yield features[batch_indices],labels[batch_indices]

def linreg(X,w,b):
    return torch.matmul(X,w)+b#不能用mv是因为w是一个[2,1]的2Dtensor

def square_loss(prediction,label):
    return (prediction-label.reshape(prediction.shape))**2/2

def sgd(params,lr,batch_size):
    with torch.no_grad():
        for param in params:
            param-=lr*param.grad/batch_size
            param.grad.zero_()

true_w=torch.tensor([2,-3.4])
print(true_w.shape)
true_b=4.2
features,labels=synthetic_data(true_w,true_b,1000)
# print(features,labels)

batch_size=10
for x,y in data_iterator(batch_size,features,labels):
    print(x,"\n",y)

# bat=[2,5,7,100,987,66]
# batt=torch.tensor(bat)
# print(features[bat].size(),features[batt].size())

w=torch.normal(0,0.01,size=(2,1),requires_grad=True)
b=torch.zeros(1,requires_grad=True)

# print(square_loss(torch.tensor([1.0,2.0,3.0]),torch.tensor([1.0,1.0,1.0])))

lr=0.01
num_epochs=10
net=linreg
loss=square_loss

for epoch in range(num_epochs):
    for X,Y in data_iterator(batch_size,features,labels):
        # print(X.shape,w.shape)
        Y_hat=net(X,w,b)
        l=loss(Y_hat,Y)
        l.sum().backward()
        sgd([w,b],lr,batch_size)
    with torch.no_grad():
        train_l=loss(net(features,w,b),labels).mean()
        print(f"epoch:{epoch+1} loss:{train_l}\n")
