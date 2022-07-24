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
#%% 初始化 模型
model=Model()
#%% 读取 模型
name="best_model2021112923"
model.load(name)
#%% 读取数据
data=np.genfromtxt('datas/sparse0_free_random.csv',delimiter=',')
# data=np.genfromtxt('datas/E_data_free_random.csv',delimiter=',')
y=polarcoordinate(np.transpose(data[:2]))
x=np.transpose(data[2:])
N=len(x)
n=int(N*0.8)
x_train=x[0:n]
x_test=x[n:N]
y_train=y[0:n]
y_test=y[n:N]

#%% 
# model.scaler_fit(x_train)

#%% 训练
# x_train=x[0:100]
# y_train=y[0:100]

es=1e-3
i=0
maxi=i
best_less:list=[None,None]
# best_model=Model()
while True:#maxi<=10000:
    maxi+=1000
    epochs=100
    # loss=model.evaluate(x_train, y_train)

    while True:
        i+=1
        print("#第%d次：" % i)
        # arr=np.arange(x_train.shape[0])
        # np.random.shuffle(arr)
        # arr=arr[0:100]
        t=time.time()
        model.fit(x_train, y_train,initial_epoch=(i-1)*epochs, epochs=i*epochs,batch_size=32,verbose=0)
        loss=model.evaluate(x_train, y_train)
        loss_te=model.evaluate(x_test, y_test) 
        if (best_less[0]==None or (loss<best_less[0]  and loss_te <best_less[1])):
            print("旧最佳模型",best_less)
            best_less=[loss,loss_te]
            print("新最佳模型",best_less)
            model.save(time.strftime("best_model%Y%m%d%H"))
        if (best_less[0]!=None and loss<best_less[0]*1.1  and loss_te <best_less[1]*0.9):
            print("旧最佳模型",best_less)
            best_less[1]=loss_te
            print("新最佳模型",best_less)
            model.save(time.strftime("best_model%Y%m%d%H"))
        if (best_less[0]!=None and loss<best_less[0]*0.9  and loss_te <best_less[1]*1.1):
            print("旧最佳模型",best_less)
            best_less[0]=loss
            print("新最佳模型",best_less)
            model.save(time.strftime("best_model%Y%m%d%H"))
        print("done. 用时%.2fs." % (time.time()-t))
        if (loss<es or i>=maxi):
            break
    if (loss<es):
        print("成功")
        break
    # print("now model:")
    # model.evaluate(x_train, y_train)
    # model.evaluate(x_test, y_test) 
    model.save(time.strftime("model%Y%m%d%H%M%S"))   
#%%  测试
 
model.evaluate(x_train, y_train)
model.evaluate(x_test, y_test)    
# %% 实验
coordinate(model.predict(np.array([[164.5985, 2370.1]]))) #-17.383 ,  -8.9591
#%% 保存模型
model.save(time.strftime("model%Y%m%d%H%M%S"))


