#%%
from model import Model
import numpy as np
import time
#%% 

def polarcoordinate(x):
    r=np.sqrt(x[:,0]**2 + x[:,1]**2)
    a=np.arccos(x[:,0]/r)*np.sign(np.sign(x[:,1])+0.5)
    y=np.zeros(x.shape)
    y[:,0]=a
    y[:,1]=r
    return y
def coordinate(I):
    x=I[:,1]*np.cos(I[:,0])
    y=I[:,1]*np.sin(I[:,0])
    o=np.zeros(I.shape)
    o[:,0]=x
    o[:,1]=y
    return o

#%%读取 模型
model=Model()
name="best_model2021111813"
model.load(name)
#%% 读输入
data=np.genfromtxt('LimitCycle_nu=0.csv',delimiter=',')
x=np.transpose(data)
print(x.shape)
#%% 计算输出并保存
ans=np.transpose(coordinate(model.predict(np.array(x))))
np.savetxt('LimitCycle_nu=0_ans.csv',ans,delimiter=',')


# %%
