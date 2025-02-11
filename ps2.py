#!/usr/bin/env python3

import math
import numpy as np
import pandas as pd

# loss
def l_bce(y, y_hat):
    return -1 * (y * math.log(y_hat) + (1 - y) * math.log(1 - y_hat))

# net input
def w_sum(x,w,b):
    return np.dot(x,w) + b

# leaky relu
def l_relu(x,alpha=0.1):
    return max(x, alpha * x)

def sigmoid(z):
    return 1 / (1 + (math.exp(-z)))

# derivative wrt y_hat
def l_bce_der(y,y_hat):
    return (y_hat - y) / ( y_hat * (1- y_hat))

# derivative sigmoid
def sigmoid_der(z):
    return sigmoid(z) * (1-sigmoid(z))

# derivative leaky relu
def l_relu_der(x,alpha=0.1):
    if x >= 0:
        return 1
    else:
        return alpha

def forward(x1,x2,w1,w2,w3,b1,b2,b3,y):
    z1 = w_sum([x1,x2],w1,b1)
    h1 = l_relu(z1)
    z2 = w_sum([x1,x2],w2,b2)
    h2 = l_relu(z2)
    z3 = w_sum([h1,h2], w3,b3)
    y_hat = sigmoid(z3)
    loss = l_bce(y,y_hat)
    df = pd.DataFrame({
        'h1': h1,
        'h2': h2,
        'z1': z1,
        'z2': z2,
        'z3': z3,
        'y_hat': y_hat,
        'loss': loss
    }, index=[0])
    print('w1', w1, 'w2', w2, 'w3', w3)
    return df

# do forward pass
x1=-1
x2=1
w1=[1,-1]
b1=0
w2=[-1,0.5]
b2=0.5
w3=[1,1]
b3=-1
y=0
df = forward(x1,x2,w1,w2,w3,b1,b2,b3,y)
print('x1:',x1,'x2:',x2,'y:',y)
print(df)

def backward(y,y_hat,z3,z2,z1,w3,b3,h2,w2,b2,h1,w1,b1,x1,x2):
    # output node
    dl_dyhat = l_bce_der(y,y_hat)
    dyhat_dz3 = sigmoid_der(z3)
    dl_dz3 = dl_dyhat * dyhat_dz3
    # weight vector 3
    # z3 = w_3[0] * h1 + w_3[1] * h2 + b3
    # tf, derivative is just h1
    dz3_dw3_1 = h1
    dl_dw3_1 = dz3_dw3_1 * dl_dz3
    dz3_dw3_2 = h2
    dl_dw3_2 = dz3_dw3_2 * dl_dz3
    # bias b3
    dz3_db3 = 1
    dl_db3 = dz3_db3 * dl_dz3
    # h1
    dz3_dh1 = w3[0]
    dl_dh1 = dz3_dh1 * dl_dz3
    #h2
    dz3_dh2 = w3[1]
    dl_dh2 = dz3_dh2 * dl_dz3
    # z2
    dh2_dz2 = l_relu_der(z2)
    dl_dz2 = dh2_dz2 * dl_dh2
    # z1
    dh1_dz1 = l_relu_der(z1)
    dl_dz1 = dh1_dz1 * dl_dh1
    # weight vector 2
    dz2_dw2_1 = x1
    dl_dw2_1 = dz2_dw2_1 * dl_dz2
    dz2_dw2_2 = x2
    dl_dw2_2 = dz2_dw2_2 * dl_dz2
    # bias 2
    dz2_db2 = 1
    dl_db2 = dz2_db2 * dl_dz2
    # weight vector 1
    dz1_dw1_1 = x1
    dl_dw1_1 = dz1_dw1_1 * dl_dz1
    dz1_dw1_2 = x2
    dl_dw1_2 = dz1_dw1_2 * dl_dz1
    # bias 1
    dz1_db1 = 1
    dl_db1 = dz1_db1 * dl_dz1
    # input x2
    dz1_dx2 = w1[1]
    dz2_dx2 = w2[1]
    dl_dx2 = (dl_dz1 * dz1_dx2) + (dl_dz2 * dz2_dx2)
    # input x1
    dz1_dx1 = w1[0]
    dz2_dx1 = w2[0]
    dl_dx1 = (dl_dz1 * dz1_dx1) + (dl_dz2 * dz2_dx1)
    # do df of derivative of loss wrt weights biases inputs and hidden layers
    df = pd.DataFrame({
        'dl_dw1_1': dl_dw1_1,
        'dl_dw1_2': dl_dw1_2,
        'dl_db1': dl_db1,
        'dl_dw2_1': dl_dw2_1,
        'dl_dw2_2': dl_dw2_2,
        'dl_db2': dl_db2,
        'dl_dw3_1': dl_dw3_1,
        'dl_dw3_2': dl_dw3_2,
        'dl_db3': dl_db3,
        'dl_dx1': dl_dx1,
        'dl_dx2': dl_dx2,
        'dl_dh1': dl_dh1,
        'dl_dh2': dl_dh2,
        'dl_dyhat': dl_dyhat,
        'dyhat_dz3': dyhat_dz3,
        'dl_dz3': dl_dz3,
        'dl_dz2': dl_dz2,
        'dl_dz1': dl_dz1
    }, index=[0])
    return df
d = backward(
    y,
    df['y_hat'][0],
    df['z3'][0],
    df['z2'][0],
    df['z1'][0],
    w3,
    b3,
    df['h2'][0],
    w2,
    b2,
    df['h1'][0],
    w1,
    b1,
    x1,
    x2
)
# round to 4 decimal places
print(d.round(4))
