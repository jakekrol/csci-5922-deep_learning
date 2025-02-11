#!/usr/bin/env python3

import math
import numpy as np
import pandas as pd

# loss
def l_bce(y, y_hat):
    return -1 * (y * math.log(y_hat) + (1 - y) * math.log(1 - y_hat))

def sigmoid(z):
    return 1 / (1 + (math.exp(-z)))

# derivative wrt y_hat
def l_bce_der(y,y_hat):
    return (y_hat - y) / ( y_hat * (1- y_hat))

# derivative sigmoid
def sigmoid_der(z):
    return sigmoid(z) * (1-sigmoid(z))

def f_h1(w1,x1,x2):
    return w1 * (x1+x2)
def f_h2(x1,x2,x3,b1):
    return (x1*x2*x3) + b1
def f_h3(x1,h1):
    return min(x1,h1)
def f_h4(w2,h1,h2):
    return w2 * (h2/h1)
def f_h5(h1,h4):
    return max(h1,h4)
def f_r(h3,h5):
    return h3 - h5

def forward(x1,x2,x3,w1,w2,b1,y):
    h1 = f_h1(w1,x1,x2)
    h2 = f_h2(x1,x2,x3,b1)
    h3 = f_h3(x1,h1)
    h4 = f_h4(w2,h1,h2)
    h5 = f_h5(h1,h4)
    r = f_r(h3,h5)
    yhat = sigmoid(r)
    loss = l_bce(y,yhat)
    df = pd.DataFrame({
        'h1': h1,
        'h2': h2,
        'h3': h3,
        'h4': h4,
        'h5': h5,
        'r': r,
        'y_hat': yhat,
        'loss': loss
    }, index=[0])
    return df
x1=2
x2=-1
x3=1
w1=1.5
w2=-3
b1=1
y=1
df_f = forward(x1,x2,x3,w1,w2,b1,y)
print(df_f)

def backward(df,x1,x2,x3,w1,w2,b1,y):
    h1 = df['h1'][0]
    h2 = df['h2'][0]
    h3 = df['h3'][0]
    h4 = df['h4'][0]
    h5 = df['h5'][0]
    r = df['r'][0]
    y_hat = df['y_hat'][0]
    loss = df['loss'][0]
    # yhat
    dl_dyhat = l_bce_der(y,y_hat)
    # r
    dyhat_dr = sigmoid_der(r)
    dl_dr = dl_dyhat * dyhat_dr
    # h5
    dr_dh5 = -1
    dl_dh5 = dl_dr * dr_dh5
    # h3
    dr_dh3 = 1
    dl_dh3 = dl_dr * dr_dh3
    # h4
    dh5_dh4 = 1 if h4 > h1 else 0
    dl_dh4 = dl_dh5 * dh5_dh4
    # w2
    dh4_dw2 = h2/h1
    dl_dw2 = dl_dh4 * dh4_dw2
    # h2
    dh4_dh2 = w2 / h1
    dl_dh2 = dl_dh4 * dh4_dh2
    # h1
    dh3_dh1 = 1 if h1 < x1 else 0
    dh5_dh1 = 1 if h1 > h4 else 0
    dh4_dh1 = w2 * h2 * -(1/h1**2)
    dl_dh1 = (dl_dh5 * dh5_dh1) + (dl_dh4 * dh4_dh1) + (dl_dh3 * dh3_dh1)
    # w1
    dh1_dw1 = x1 + x2
    dl_dw1 = dl_dh1 * dh1_dw1
    # b1
    dh2_db1 = 1
    dl_db1 = dl_dh2 * dh2_db1
    # x1
    dh3_dx1 = 1 if x1 < h1 else 0
    dh2_dx1 = x2 * x3
    dh1_dx1 = w1
    dl_dx1 = (dl_dh3 * dh3_dx1) + (dl_dh2 * dh2_dx1) + (dl_dh1 * dh1_dx1)
    # x2
    dh2_dx2 = x1 * x3
    dh1_dx2 = w1
    dl_dx2 = (dl_dh2 * dh2_dx2) + (dl_dh1 * dh1_dx2)
    # x3
    dh2_dx3 = x1 * x2
    dl_dx3 = dl_dh2 * dh2_dx3
    # return all gradients in dictionary
    # include hidden layers, r, and yhat, too
    gradients = {
        'dl_dw1': dl_dw1,
        'dl_dw2': dl_dw2,
        'dl_db1': dl_db1,
        'dl_dx1': dl_dx1,
        'dl_dx2': dl_dx2,
        'dl_dx3': dl_dx3,
        'dl_dh1': dl_dh1,
        'dl_dh2': dl_dh2,
        'dl_dh3': dl_dh3,
        'dl_dh4': dl_dh4,
        'dl_dh5': dl_dh5,
        'dl_dr': dl_dr,
        'dl_dyhat': dl_dyhat,
        'dyhat_dr': dyhat_dr
    }
    return gradients
d_b = backward(df_f,x1,x2,x3,w1,w2,b1,y)
print(d_b)