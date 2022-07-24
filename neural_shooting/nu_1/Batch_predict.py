#%%
from model import Model
import numpy as np
import time

#%%读取 模型
model=Model()
name="best_model2022071415"
model.load(name)
#%% 读输入
data=np.genfromtxt('datas/LimitCycle_nu=0.1.csv',delimiter=',')
x=np.transpose(data)
print(x.shape)
#%% 计算输出并保存
ans=np.transpose(model.predict(np.array(x)))
np.savetxt('LimitCycle_nu=0.1_ans.csv',ans,delimiter=',')


# %%
