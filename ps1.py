#!/usr/bin/env python3

import math, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

w1=1
w2=1
b=-1
eta = 0.25

def output(x1,x2,w1,w2,b):
    return np.sign( (w1*x1) + (w2*x2) + b)

def delta_w(eta, target, output, feature):
    return eta * (target-output) * feature
def w_p(w,delta_w):
    return w + delta_w

train = pd.DataFrame({
    'sample': [1,2,3,4],
    'x1': [1,-1,-1,1],
    'x2':[1,1,-1,-1],
    'y':[-1,1,1,-1]
    }
)
print(train)
# loop over rows, compute output and update weights
# fill a dataframe with updated weights
df = pd.DataFrame(columns=['w1','w2','b'])
for index, row in train.iterrows():
    x1 = row['x1']
    x2 = row['x2']
    y = row['y']
    o = output(x1,x2,w1,w2,b)
    w1 = w_p(w1, delta_w(eta, y, o, x1))
    w2 = w_p(w2, delta_w(eta, y, o, x2))
    b = w_p(b, delta_w(eta, y, o, 1))
    df = pd.concat([df, pd.DataFrame({'w1':[w1], 'w2':[w2], 'b':[b]}) ])

print(df)

test = pd.DataFrame({
    'sample': [1,2,3,4],
    'x1': [0.5,2,-0.5,-0.5],
    'x2':[2,2,0,-2],
    'y':[1,-1,1,-1]
    }
)

# q3b
# plot data points
for index, row in test.iterrows():
    x1 = row['x1']
    x2 = row['x2']
    y = row['y']
    plt.scatter(x1, x2, marker='x' if y == -1 else 'o', color = 'black')
# solve for x2 to plot decision boundary
def f(x1, w1, w2, b):
    return  ((-w1 * x1) - b) / w2
x1 = np.linspace(-2.5, 2.5, 100)  # Adjust range as needed
x2 = []
for i in x1:
    x2.append(f(i, w1, w2, b))
plt.plot(x1, x2, label="Decision Boundary", color = 'black')

# Plot formatting
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("Decision Boundary and test data points (x: $y=-1$, o: $y=1$)")
plt.legend()
plt.savefig("ps1-q3b.png")

# compute output for test data
print("Output for test data")
print("Trained model parameters: w1, w2, b", w1, w2, b)
for index, row in test.iterrows():
    x1 = row['x1']
    x2 = row['x2']
    y = row['y']
    o = output(x1,x2,w1,w2,b)
    print(f"Output for ({x1},{x2}): {o}")

plt.plot(x1, x2, label="Decision Boundary", color = 'black')

# q3e extra credit
# # plot data points
# for index, row in train.iterrows():
#     x1 = row['x1']
#     x2 = row['x2']
#     y = row['y']
#     plt.scatter(x1, x2, marker='x' if y == -1 else 'o', color = 'red')
# for index, row in test.iterrows():
#     x1 = row['x1']
#     x2 = row['x2']
#     y = row['y']
#     plt.scatter(x1, x2, marker='x' if y == -1 else 'o', color = 'black')

# # solve for x2 to plot decision boundary
# def f(x1, w1, w2, b):
#     return  ((-w1 * x1) - b) / w2

# # updated decision boundary
# x1 = np.linspace(-2.5, 2.5, 100)  # Adjust range as needed
# x2 = []
# for i in x1:
#     x2.append(f(i, w1, w2, b))
# plt.plot(x1, x2, label="Decision Boundary post-training", color = 'black')

# # initial decision boundary
# w1=1
# w2=1
# b=-1
# x1 = np.linspace(-2.5, 2.5, 100)  # Adjust range as needed
# x2 = []
# for i in x1:
#     x2.append(f(i, w1, w2, b))
# plt.plot(x1, x2, label="Decision Boundary initial", color = 'red', linestyle='--')
# # Plot formatting
# plt.axhline(0, color='black', linewidth=0.5)
# plt.axvline(0, color='black', linewidth=0.5)
# plt.grid(True)
# plt.xlabel("$x_1$")
# plt.ylabel("$x_2$")
# plt.title("Extra credit: decision boundary comparison")
# plt.legend(loc='upper left')
# plt.savefig("ps1-q3e.png")