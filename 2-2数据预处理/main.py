import torch
import torch.nn as nn
import os
import pandas as pd

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

os.makedirs(os.path.join("..","data"),exist_ok=True)
file=os.path.join("..","data","tinyhouse.csv")
with open(file,"w") as f:
    f.write("numrooms,alley,price\n")
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

data=pd.read_csv(file)
print(data)

missing_values=data.isnull().sum()
mostna_idx=missing_values.idxmax()
print(mostna_idx)
result=data.drop(mostna_idx,axis=1)

result=result.fillna(result.mean())
print(result)
result=result.values
result_tensor=torch.tensor(result)
print(result_tensor)