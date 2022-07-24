#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np 
import time
import matplotlib.pyplot as plt
# import pickle5 as pickle


# In[2]:


def neural_net(X, weights, biases):
    num_layers = len(weights) + 1  
    H=X
#    H = 2.0*(X -  X.min(0))/( X.max(0) -  X.min(0)) - 1.0
    for l in range(0,num_layers-2):
        W = weights[l]
        b = biases[l]
        H = tf.nn.tanh(tf.add(tf.matmul(H, W), b))
    W = weights[-1]
    b = biases[-1]
    Y =  tf.exp(tf.add(tf.matmul(H, W), b))
    return Y


def xavier_init(size):
    in_dim = size[0]
    out_dim = size[1]        
    xavier_stddev = np.sqrt(2.0/(in_dim + out_dim))
    return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev,dtype=tf.float64,seed=0), dtype=tf.float64)


# In[3]:


def neural_net_u(X, weights, biases):
    u= neural_net(X, weights, biases)
    return u

def neural_net_v(X, weights, biases):
    v= neural_net(X, weights, biases)
    return v


# In[4]:


def drift(x,y,mu,b,theta,c_x,c_p,nu,y0,gama,f_0,c_f,beta):
    g1=-(f_0*x**beta*(y0 - y + mu*(nu - theta*(x**gama/(c_x**gama + x**gama) - 1) + (b*x**gama)/(c_p**gama + x**gama) - 1)))/(c_f**beta + x**beta)
    g2= y0 - y + mu*(nu - theta*(x**gama/(c_x**gama + x**gama) - 1) - (b*x**gama)/(c_p**gama + x**gama) + 1)
    return g1,g2

def diffusion(u,eps,f_0,c_f,beta):
    s=(f_0*eps*u**beta)/(c_f**beta + u**beta)
    return s


def net_f(tf1,weights_u,biases_u,weights_v,biases_v,mu,b,theta,c_x,c_p,nu,y0,gama,f_0,c_f,beta,eps):
    u=neural_net(tf1, weights_u, biases_u)
    v=neural_net(tf1, weights_v, biases_v)
    u_t = tf.gradients(u, tf1)[0]
    u_tt = tf.gradients(u_t, tf1)[0]
    v_t = tf.gradients(v, tf1)[0]
    v_tt = tf.gradients(v_t, tf1)[0]
    
    g1,g2=drift(u,v,mu,b,theta,c_x,c_p,nu,y0,gama,f_0,c_f,beta)
    g1_u=tf.gradients(g1, u)[0]
    g1_v=tf.gradients(g1, v)[0]
    g1_uu=tf.gradients(g1_u, u)[0]
    g1_uv=tf.gradients(g1_v, u)[0]
    
    g2_u=tf.gradients(g2, u)[0]
    g2_v=-1
    g2_vv=0
    g2_uv=0
    
    s=diffusion(u,eps,f_0,c_f,beta)
    s_u=tf.gradients(s, u)[0]
    s_uu=tf.gradients(s_u, u)[0]
    s_uuu=tf.gradients(s_uu, u)[0]
    

    return -u_tt+u_t*(g1_u+(eps*mu)**2/2*(s_uu*s+s_u**2))+v_t*g1_v+2*s_u/s*u_t*(u_t-g1-(eps*mu)**2/2*s_u*s)-s_u/s*(u_t-g1-(eps*mu)**2/2*s_u*s)**2-(u_t-g1-(eps*mu)**2/2*s_u*s)*(g1_u+(eps*mu)**2/2*(s_uu*s+s_u**2))-s**2*(v_t-g2)*g2_u+(eps*mu)**2*s**2/2*(g1_uu+(eps*mu)**2/2*s_uuu*s+3*(eps*mu)**2/2*s_uu*s_u+g2_uv-s_u/s*(g1_u+(eps*mu)**2/2*(s_uu*s+s_u**2))-(s_uu*s-s_u**2/(s**2))*(g1+(eps*mu)**2/2*s*s_u)),-v_tt+g2_u*u_t+g2_v*v_t-1/s**2*(u_t-g1-(eps*mu)**2/2*s_u*s)*g1_v-(v_t-g2)*g2_v+0.5*(eps*mu)**2*(g1_uv+g2_vv-g1_v*s_u/s)

# In[8]:

def fun(u_R,v_R,index1):
    layers = [1] + 4* [20] + [1]
    L = len(layers)
    #tt1=time.time()
    np.random.seed(0)
    
    weights_u = [xavier_init([layers[l], layers[l+1]]) for l in range(0, L-1)]    
    biases_u = [tf.Variable( tf.zeros((1, layers[l+1]),dtype=tf.float64)) for l in range(0, L-1)]
    
    weights_v = [xavier_init([layers[l], layers[l+1]]) for l in range(0, L-1)]    
    biases_v = [tf.Variable( tf.zeros((1, layers[l+1]),dtype=tf.float64)) for l in range(0, L-1)]
    
    
    # In[9]:
    
    eps=tf.Variable([.2],dtype=tf.float64,trainable=False)
    mu=tf.Variable([250],dtype=tf.float64,trainable=False)
    b=tf.Variable([4],dtype=tf.float64,trainable=False)
    theta=tf.Variable([5],dtype=tf.float64,trainable=False)
    c_x=tf.Variable([58],dtype=tf.float64,trainable=False)
    c_p=tf.Variable([110],dtype=tf.float64,trainable=False)
    nu=tf.Variable([0],dtype=tf.float64,trainable=False)
    y0=tf.Variable([2000],dtype=tf.float64,trainable=False)
    gama=tf.Variable([4],dtype=tf.float64,trainable=False)
    f_0=tf.Variable([0.694],dtype=tf.float64,trainable=False)
    c_f=tf.Variable([43.9],dtype=tf.float64,trainable=False)
    beta=tf.Variable([2.0],dtype=tf.float64,trainable=False)
    t0=0
    index=index1+185
    t1=index*0.05+1
    Nt=500
    vect=np.linspace(t0,t1,Nt+1)[:,None]
    tf1_tf = tf.to_double(vect) 
    
    
    # In[10]:
    
    
    uL_nn=neural_net(tf1_tf, weights_u, biases_u)[0]
    uR_nn=neural_net(tf1_tf, weights_u, biases_u)[-1]
    vL_nn=neural_net(tf1_tf, weights_v, biases_v)[0]
    vR_nn=neural_net(tf1_tf, weights_v, biases_v)[-1]
    u_L = (b-1)**(-1/gama)*c_p;
    v_L = y0+mu*(theta+nu-theta*c_p**gama/((b-1)*c_x**gama+c_p**gama));
    
    # In[11]:
    
    
    loss_u=tf.reduce_mean(tf.square(uL_nn-u_L))+tf.reduce_mean(tf.square(uR_nn-u_R))
    loss_v=tf.reduce_mean(tf.square(vL_nn-v_L))+tf.reduce_mean(tf.square(vR_nn-v_R))
    f_pred = net_f(tf1_tf,weights_u,biases_u,weights_v,biases_v,mu,b,theta,c_x,c_p,nu,y0,gama,f_0,c_f,beta,eps)
    
    
    # In[12]:
    
    
    loss_ode1 = tf.reduce_mean(tf.square(f_pred[0]))
    loss_ode2=tf.reduce_mean(tf.square(f_pred[1]))
    
    # loss=loss_ode1 +600*loss_u+loss_ode2 +300*loss_v
    loss=10*loss_ode1 + 30*loss_u + loss_ode2 + 5*loss_v
    
    
    # In[13]:
    
    
    optimizer_Adam = tf.train.AdamOptimizer(1e-3)
    train_op_Adam = optimizer_Adam.minimize(loss)
    
    loss_record = []
    loss_ode1_record = []
    loss_ode2_record = []
    loss_u_record = []
    loss_v_record = []
    sigma1_record = []
    sigma2_record = []
    mu_record = []
    b_record = []
    theta_record = []
    c_x_record = []
    c_p_record = []
    nu_record = []
    y0_record = []
    gama_record = []
    f_0_record = []
    c_f_record = []
    beta_record = []
    saver = tf.train.Saver(max_to_keep=1000)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    min_loss = 1e16
    
    
    # In[ ]:
    
    
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        u_pred=neural_net(tf1_tf,weights_u,biases_u) 
        v_pred=neural_net(tf1_tf,weights_v,biases_v) 
        for i in range(400001):
    
            sess.run(train_op_Adam )
            if i % 500 == 0:
                (loss_result, loss_ode1_result, loss_u_result , loss_ode2_result, loss_v_result ) = sess.run([loss,
                        loss_ode1, loss_u,loss_ode2, loss_v])
                ut0= sess.run(u_pred)
                vt0= sess.run(v_pred)
                (temp_mu,temp_b,temp_theta,temp_c_x,temp_c_p,temp_nu,temp_y0,temp_gama,temp_f_0,temp_c_f,temp_beta,temp_eps)=sess.run([mu,b,theta,c_x,c_p,nu,y0,gama,f_0,c_f,beta,eps])
                loss_record.append(loss_result)
                loss_ode1_record.append(loss_ode1_result)
                loss_u_record.append(loss_u_result)
                loss_ode2_record.append(loss_ode2_result)
                loss_v_record.append(loss_v_result)
     #           eps_record.append(temp_eps)
                mu_record.append(temp_mu)
                b_record.append(temp_b)
                theta_record.append(temp_theta)
                c_x_record.append(temp_c_x)
                c_p_record.append(temp_c_p)
                nu_record.append(temp_nu)
                y0_record.append(temp_y0)
                gama_record.append(temp_gama)
                f_0_record.append(temp_f_0)
                c_f_record.append(temp_c_f)
                beta_record.append(temp_beta)
                if loss_result<min_loss:
                    min_loss=loss_result
                    u_opt= sess.run(u_pred) 
                    v_opt= sess.run(v_pred)
                    i_opt=i
                print ('  %d  %8.2e  %8.2e %8.2e  %8.2e %8.2e ' % (i, loss_result,loss_ode1_result,loss_ode2_result, loss_u_result, loss_v_result) )
    
             # if i % 50000 == 0:
                # (weights_u_np,biases_u_np,weights_v_np,biases_v_np )=sess.run([weights_u,biases_u,weights_v,biases_v ])
                # sample_list = {"weights_u": weights_u_np, "biases_u": biases_u_np,"weights_v": weights_v_np, "biases_v": biases_v_np}
                # file_name = './result/hyper' + str(i) + '.pkl'
                # open_file = open(file_name, "wb")
                # open_file.close()              
                # np.savetxt('./result/loss.txt',np.array(loss_record),fmt='%10.5e')
                # np.savetxt('./result/loss_ode1.txt',np.array(loss_ode1_record),fmt='%10.5e')
                # np.savetxt('./result/loss_u.txt',np.array(loss_u_record),fmt='%10.5e')
                # np.savetxt('./result/loss_ode2.txt',np.array(loss_ode2_record),fmt='%10.5e')
                # np.savetxt('./result/loss_v.txt',np.array(loss_v_record),fmt='%10.5e')
                # np.savetxt('./result/loss_u' + str(index) + '.txt',np.array(ut0),fmt='%10.5e')
                # np.savetxt('./result/loss_v' + str(index) + '.txt',np.array(vt0),fmt='%10.5e')           
                # np.savetxt('./result/u_opt' + str(index) + '.txt',np.array(u_opt),fmt='%10.5e')
                # np.savetxt('./result/v_opt' + str(index) + '.txt',np.array(v_opt),fmt='%10.5e')
            if i % 50000 == 0:
                (weights_u_np,biases_u_np,weights_v_np,biases_v_np )=sess.run([weights_u,biases_u,weights_v,biases_v ])
                sample_list = {"weights_u": weights_u_np, "biases_u": biases_u_np,"weights_v": weights_v_np, "biases_v": biases_v_np}
                file_name = './result/hyper' + str(i) + '.pkl'
                open_file = open(file_name, "wb")
                open_file.close()              
                np.savetxt('./result/loss' + str(index) + '.txt',np.array(loss_record),fmt='%10.5e')
                # np.savetxt('./result/loss_ode1.txt',np.array(loss_ode1_record),fmt='%10.5e')
                # np.savetxt('./result/loss_u.txt',np.array(loss_u_record),fmt='%10.5e')
                # np.savetxt('./result/loss_ode2.txt',np.array(loss_ode2_record),fmt='%10.5e')
                # np.savetxt('./result/loss_v.txt',np.array(loss_v_record),fmt='%10.5e')
                # np.savetxt('./result/loss_u.txt',np.array(ut0),fmt='%10.5e')
                # np.savetxt('./result/loss_v.txt',np.array(vt0),fmt='%10.5e')           
                np.savetxt('./result/u_opt' + str(index) + '.txt',np.array(u_opt),fmt='%10.5e')
                np.savetxt('./result/v_opt' + str(index) + '.txt',np.array(v_opt),fmt='%10.5e')


# In[]
data = np.loadtxt('LimitCycle_nu=0.txt')
u = data[0,1722]
v = data[1,1722]
# u = data[0,2290]
# v = data[1,2290]
# u = data[0,3326]
# v = data[1,3326]
L = 100
# i = 60
# fun(u,v,i)
for i in range(L):
    fun(u,v,i)
# In[ ]:


# plt.plot(ut0,vt0,'r')
# plt.plot(u_opt,v_opt,'g.')
# u_L=83.1841
# u_R=19.4883
# v_L=2234.6638
# v_R=2556.0621


# In[29]:


# plt.plot(ut0,'r')
# plt.plot(u_opt,'g--')


# In[30]:


# plt.plot(vt0,'r')
# plt.plot(v_opt,'g--')


# In[31]:


# import numpy as np
# u_opt=np.loadtxt('./result/u_opt-mat.txt')
# v_opt=np.loadtxt('./result/v_opt-mat.txt')
# plt.plot(u_opt,v_opt)


# In[ ]:




    
