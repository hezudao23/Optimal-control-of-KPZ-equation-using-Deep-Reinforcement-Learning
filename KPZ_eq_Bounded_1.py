# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 10:42:20 2018

@author: yhuang16
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 23:51:40 2018

@author: yunlo
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 11:20:33 2018

@author: yunlo
"""

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
tf.reset_default_graph()


sess = tf.InteractiveSession()
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 1e-2
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 500, 0.9, staircase=True)




dim = 10
batch_size = 1
""" Extra 1 is for time stamp"""
total_time = 1
num_time_interval = 10 # this cannot be very large 10
delta_time = total_time / num_time_interval
nu = 1
lda = 0.01
sample_size = 128
#num_hidden = [dim, dim + 10, dim + 10, dim]
num_hidden = [dim +1 , dim + 10 + 1, dim + 10 + 1, dim]
''' This is the lambda'''
#####
'''Some constant for control signal '''
Mg = tf.constant([2.5]) # 4.5
T_0 = tf.constant([3.0]) # 5 
#lr = 1e-2
####

def z_init():
#    return tf.Variable(tf.ones((1, dim)), trainable = False) 
    return tf.Variable(tf.ones((1, dim))*0.9, dtype = tf.float32, trainable = False)
def A_init():
    # component 1
    C_1 = tf.ones((dim, dim))
    
    C_1 =  tf.matrix_band_part(C_1, 1, 1) 
    # component 2
    C_2 =  tf.diag( 3 * tf.ones((1, dim))[0] )
    
    C_2 = tf.subtract(C_1,  C_2)
    """ Till now, we have the diagonal banded matrix already, we need two extra term """
    indices_1 = [[0, dim - 1]]  # A list of coordinates to update.
    indices_2 = [[dim - 1, 0]]  # A list of coordinates to update.
    
    values = [1.0]  # A list of values corresponding to the respective
                # coordinate in indices.

    shape = [dim, dim]  # The shape of the corresponding dense tensor, same as `c`.

    delta_1 = tf.SparseTensor(indices_1, values, shape)
    delta_2 = tf.SparseTensor(indices_2, values, shape)
    
    return C_2 + tf.sparse_tensor_to_dense(delta_1) + tf.sparse_tensor_to_dense(delta_2)
    


''' Here I tried to change the optimal control to the strict bounded version, ...
compared to the old smooth approximation version. '''
def control(u):
    C_1 = tf.maximum(u, -1 * Mg)
    return tf.minimum(C_1, Mg)

#def noise():
#    return tf.random_normal(shape = (1, dim), stddev=np.sqrt(delta_time))
'''As a input u is a vector with dimension (1, dim)''' 
'''We woudl like to have the z in dimesion (1, dim) '''

z = z_init()
A = A_init()
y_init = tf.Variable([1.0], dtype = tf.float32)
y = tf.Variable([0.0], dtype = tf.float32, trainable= False)
'''y is the value function '''


def subnetwork(z, t_idx):
    input_= tf.concat([z, t_idx], 1)
    for i in range(1, len(num_hidden)):
        name_1 = 'layer_{}'.format(i)
        with tf.variable_scope(name_1, reuse=tf.AUTO_REUSE):
            W = tf.get_variable('Matrix', [num_hidden[i-1], num_hidden[i]], tf.float32, tf.random_normal_initializer(stddev = 1/np.sqrt(num_hidden[i - 1] + num_hidden[i])))
           # W = tf.get_variable('Matrix', [num_hidden[i-1], num_hidden[i]], tf.float32, tf.random_normal_initializer(stddev = 0.1)) # 0.01
            b = tf.get_variable('Bias', [1, num_hidden[i]], tf.float32,  tf.random_normal_initializer(stddev = 1/np.sqrt(num_hidden[i])))
            if i < len(num_hidden) - 1:
#                input_ = tf.nn.tanh(tf.add(tf.matmul(input_, W), b))
                input_ = tf.nn.softplus(tf.add(tf.matmul(input_, W), b))
            else:
                input_ = tf.add(tf.matmul(input_, W), b)
    
    return input_




def lag(z, c_signal):
    z_mean = tf.reduce_mean(z)
    c_1 = tf.subtract(z, z_mean)
    c_2 = lda * tf.reduce_sum(tf.square(c_signal))
    res = tf.add(tf.reduce_mean(tf.square(c_1)), c_2)
    return res

def bc(z):
    z_mean = tf.reduce_mean(z)
    c_1 = tf.subtract(z, z_mean)
    res = tf.reduce_mean(tf.square(c_1))
    return res


noise_collection = tf.placeholder(tf.float32, shape = [batch_size, num_time_interval, dim])
loss_sum = 0

for epoch in range(batch_size):
    noise = noise_collection[epoch]
    z = z_init()
    for idx in range(1, num_time_interval + 1):
        #name = 'step_{}'.format(idx)
        #grad_y = subnetwork(z, name)
        
        idx_ = tf.constant([[float(idx)]])
        grad_y = subnetwork(z, idx_)
        hessian = tf.stack([tf.gradients(grad_y[:, idy], z)[0] for idy in range(dim)], axis=1)[0]
        hessian_part = tf.reshape(tf.diag_part(hessian), [1, dim])
        c_signal = -0.5/lda * tf.multiply(tf.square(z), hessian_part) 
        c_signal = control(c_signal)
        u_new =  tf.sqrt(tf.add(T_0, c_signal))
    #    u_new =  tf.sqrt(T_0)
        D_1 = tf.multiply(u_new, z)
        D = tf.diag(D_1[0])
        B = tf.slice(noise, [idx - 1, 0], [1, dim])
    #    z = tf.add(z, tf.add(delta_time * tf.matmul(z, A), tf.matmul(B, D)))
        if idx == 1:
            y = tf.add(y_init, tf.add(-delta_time * lag(z, c_signal), tf.matmul(grad_y, tf.matmul(D, tf.transpose(B)))))
        else:
            y = tf.add(y, tf.add(-delta_time * lag(z, c_signal), tf.matmul(grad_y, tf.matmul(D, tf.transpose(B)))))
        z = tf.add(z, tf.add(delta_time * tf.matmul(z, A), tf.matmul(B, D)))
         
    term_res = bc(z)
    loss = tf.nn.l2_loss(tf.subtract(term_res, y))
    loss_sum = loss_sum + loss / batch_size
    
    
''' Here we would like to update the parameter explicitly '''





#train_step = tf.train.RMSPropOptimizer(lr).minimize(loss_sum)
#train_step = tf.train.AdamOptimizer(lr).minimize(loss_sum)   
#
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_sum, global_step=global_step)  

sess.run(tf.global_variables_initializer())

noise_S = []

for idx in range(sample_size):
    
    noise_S.append(np.random.randn(num_time_interval, dim) * np.sqrt(delta_time))



loss_mem = []
y_init_mem = []
iter_mem = []
  
for i in range(20000 + 1):
    noise_cur = []
    for j in range(batch_size):
#        seed = int(np.random.uniform(0, sample_size - 1, 1))
        seed = i % sample_size
        noise_cur.append(noise_S[seed])
    
    
    train_step.run(feed_dict = {noise_collection: noise_cur})
    
    if i % 10 == 0:
        [loss_res, y_init_res] = sess.run([loss_sum, y_init], feed_dict = {noise_collection: noise_cur})
        print('Step %d, loss is %f and y_init is %f' % (i, loss_res, y_init_res))
        loss_mem.append(loss_res)
        y_init_mem.append(y_init_res[0])
        iter_mem.append(i)
    
    del noise_cur     