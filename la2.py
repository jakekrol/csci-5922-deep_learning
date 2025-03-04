#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
import sys
# import dataloader
from torch.utils.data import DataLoader
from sklearn.utils import resample
from torch.optim.lr_scheduler import StepLR

# CONST
TRAIN_PROP = 0.25
EPOCHS=300
DROPOUT=True
NAME='dropout'
LR_DECAY=False
MOMENTUM=None
torch.manual_seed(42)

print("CONSTANTS")
print(f"TRAIN_PROP: {TRAIN_PROP}")
print(f"EPOCHS: {EPOCHS}")
print(f"DROPOUT: {DROPOUT}")
print(f"NAME: {NAME}")
print(f"LR_DECAY: {LR_DECAY}")
print(f"MOMENTUM: {MOMENTUM}")


# df = pd.read_csv('host_train.csv')
# df = df.dropna().reset_index(drop=True)
# print(df.head())

# # encoding the categorical columns
# col_conv = [
#     'Department',
#     'Ward_Facility',
#     'Ward_Type',
#     'Type of Admission',
#     'Illness_Severity',
#     'Age',
#     'Stay_Days'
# ]
# file='lab_assignment2.categories.txt'
# try:
#     os.remove(file)
# except FileNotFoundError:
#     pass
# for col in col_conv:
#     label_encoder = LabelEncoder()
#     df.loc[:,col] = label_encoder.fit_transform(df.loc[:,col])
#     for i in range(len(label_encoder.classes_)):
#         with open(file, 'a') as f:
#             f.write(f'{col},{i},{label_encoder.classes_[i]}\n')
# df.to_csv('host_train_encoded.tsv', index=False,sep='\t')
# df = df.drop(columns=['case_id','patientid'])

# # one-hot encode nominal
# onehot = True
# if onehot:
#     col_1hot = ['Hospital', 'Hospital_type', 'Hospital_city', 'Hospital_region', 'Department', 'Ward_Type', 'Ward_Facility', 'City_Code_Patient', 'Illness_Severity', 'Age']
#     df = pd.get_dummies(df, columns=col_1hot, dtype=np.int64)
#     df.to_csv('host_train_encoded_onehot.tsv', index=False,sep='\t')

# # split
df = pd.read_csv('host_train_encoded_onehot.tsv', sep='\t')
X = df.drop(columns=['Stay_Days'])
y = df['Stay_Days']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42, shuffle=True)
X_train = X_train.reset_index(drop=True)
X_test.reset_index(drop=True)
y_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

if TRAIN_PROP < 1:
    print(f'Subsetting to {int(TRAIN_PROP * 100)}%')
    print('Before', y_train.shape)
    y_train, _ = train_test_split(y_train, test_size=1-TRAIN_PROP, stratify=y_train, random_state=42)
    print('After', y_train.shape)
    X_train = X_train.iloc[y_train.index,:]
    print('After', X_train.shape)

# scale continuous columns
file = f'lab_assignment2.train_scalers.{TRAIN_PROP}.txt'
try:
    os.remove(file)
except FileNotFoundError:
    pass
col_cont = ['Available_Extra_Rooms_in_Hospital', 'Patient_Visitors', 'Admission_Deposit']
d = {}
for col in col_cont:
    print(col)
    mu = X_train.loc[:,col].mean()
    sigma = X_train.loc[:,col].std()
    X_train.loc[:,col] = (X_train.loc[:,col] - sigma) / mu
    d[col] = [mu, sigma]
    with open(file, 'a') as f:
        f.write(f'{col},{mu},{sigma}\n')
X_train.to_csv(f'covid_X_train_{TRAIN_PROP}.tsv', index=False,sep='\t')

# scale test data
X_test.loc[:,'Available_Extra_Rooms_in_Hospital'] = \
    (X_test.loc[:,'Available_Extra_Rooms_in_Hospital'] - \
        d['Available_Extra_Rooms_in_Hospital'][0]) / \
            d['Available_Extra_Rooms_in_Hospital'][1]
X_test.loc[:,'Patient_Visitors'] = \
    (X_test.loc[:,'Patient_Visitors'] - \
        d['Patient_Visitors'][0]) / \
            d['Patient_Visitors'][1]
X_test.loc[:,'Admission_Deposit'] = \
    (X_test.loc[:,'Admission_Deposit'] - \
        d['Admission_Deposit'][0]) / \
            d['Admission_Deposit'][1]
# X_test depends on train prop since we scale based on train data
X_test.to_csv(f'covid_X_test_{TRAIN_PROP}.tsv', index=False,sep='\t')
y_train.to_csv(f'covid_y_train_{TRAIN_PROP}.tsv', index=False,sep='\t')
# y test is invariant to train prop
y_test.to_csv(f'covid_y_test.tsv', index=False,sep='\t')

### network
if DROPOUT:
    p_drop=0.8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# build MLP with 5 hidden layers and ReLU activation
class FFN(nn.Module):
    def __init__(self, input_size, output_size):
        super(FFN, self).__init__()
        hidden_size = 64
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU() 
        if DROPOUT:
            self.drop1 = nn.Dropout(p=p_drop)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        if DROPOUT:
            self.drop2 = nn.Dropout(p=p_drop)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        if DROPOUT:
            self.drop3 = nn.Dropout(p=p_drop)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.relu4 = nn.ReLU()
        if DROPOUT:
            self.drop4 = nn.Dropout(p=p_drop)
        self.fc5 = nn.Linear(hidden_size, output_size)
        self.relu5 = nn.ReLU()
        # Apply Xavier initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.fc5(x)
        x = self.relu5(x)
        return x
# input_size = 15
input_size = 125
num_classes = 11 

model = FFN(input_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()  # Automatically applies softmax
if MOMENTUM:
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=MOMENTUM)
else:
    optimizer = optim.SGD(model.parameters(), lr=0.001)

### minimal example
# Load data
# X_train = torch.tensor(pd.read_csv('covid_X_train.tsv', sep='\t').iloc[0,:].values.reshape(1,15), dtype=torch.float32)
# y_train = torch.tensor(pd.read_csv('covid_y_train.tsv', sep='\t').iloc[0,:].values, dtype=torch.long)
# print('X_train')
# print(X_train)
# print('y_train')
# print(y_train)

# # Forward pass
# print('Forward pass')
# outputs = model(X_train)  # Raw logits
# print("Logits:", outputs)  # Raw scores (not probabilities)
# loss = criterion(outputs, y_train)  # No need for softmax
# loss.backward()
# optimizer.step()
# print("Logits:", outputs)  # Raw scores (not probabilities)
# print("Predicted class:", torch.argmax(outputs, dim=1))  # Predicted class
# print("Ground truth:", y_train[0])
# sys.exit()

# load data
# train
X_train = pd.read_csv(f'covid_X_train_{TRAIN_PROP}.tsv', sep='\t')
X_train = X_train.dropna()
idx = X_train.index.values
y_train = pd.read_csv(f'covid_y_train_{TRAIN_PROP}.tsv', sep='\t')
y_train = y_train.iloc[idx,:]
X_train = torch.tensor(X_train.values, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train.values, dtype=torch.long).to(device)
# test
X_test = pd.read_csv(f'covid_X_test_{TRAIN_PROP}.tsv', sep='\t')
X_test = X_test.dropna()
idx = X_test.index.values
y_test = pd.read_csv(f'covid_y_test.tsv', sep='\t')
y_test = y_test.iloc[idx,:]
X_test = torch.tensor(X_test.values, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test.values, dtype=torch.long).to(device)
tensor_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = DataLoader(tensor_dataset, batch_size=1024, shuffle=True)

# Start training
if LR_DECAY: 
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
epochs = [] # x axis for histry
losses = [] # loss histry
accuracies = [] # acc histry
accuracies_test = [] # test acc histry
for epoch in range(EPOCHS):
    epoch_loss = 0
    correct = 0
    correct_test = 0
    total = 0
    total_test = 0
    for i, (x, y) in enumerate(train_loader):
        if torch.isnan(x).any() or torch.isnan(y).any():
            print(f"NaN found in data at batch {i}")
        optimizer.zero_grad()  # Reset gradients
        o = model(x)  # Forward pass
        y = y.squeeze()
        loss = criterion(o, y)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        
        # Accumulate loss and accuracy
        epoch_loss += loss.item()
        yhat = torch.argmax(o, dim=1)
        correct += (yhat == y).sum().item()
        total += y.size(0)

        # test accuracy
        o = model(X_test)
        y = y_test.squeeze()
        yhat = torch.argmax(o, dim=1)
        correct_test += (yhat == y).sum().item()
        total_test += y.size(0)


    # Print stats
    epoch_loss /= len(train_loader)
    accuracy = correct / total
    accuracy_test = correct_test / total_test
    print(f"Epoch {epoch+1}/{EPOCHS}, loss: {epoch_loss:.4f}, acc: {accuracy:.4f}, test acc: {accuracy_test:.4f}")
    epochs.append(epoch)
    losses.append(epoch_loss)
    accuracies.append(accuracy)
    accuracies_test.append(accuracy_test)
    # decay learning rate
    if LR_DECAY:
        scheduler.step()
    
    
# Plot loss and accuracy
fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(epochs, losses, label='Loss')
ax1.set_ylabel('Loss')
ax2.plot(epochs, accuracies, label='Train accuracy')
ax2.plot(epochs, accuracies_test, label='Test Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_ylim([0,1])
ax2.legend()
try:
    fig.suptitle(f'{NAME} train {int(TRAIN_PROP * 100)}%')
    plt.savefig(f'p_{NAME}_{TRAIN_PROP}.png')
except:
    plt.savefig(f'p_{NAME}_{TRAIN_PROP}.png')


# compute average precision and accuracy across all classes for trained model
# on test data
# o = model(X_test)
# y = y_test.squeeze()
# yhat = torch.argmax(o, dim=1)
# correct = (yhat == y).sum().item()
# total = y.size(0)
# accuracy = correct / total
# print(f"Test accuracy: {accuracy:.4f}")
# # confusion matrix
# confusion = torch.zeros(num_classes, num_classes)
# for i in range(y.size(0)):
#     confusion[y[i], yhat[i]] += 1
# print(confusion)
# # precision
# precision = torch.zeros(num_classes)
# for i in range(num_classes):
#     precision[i] = confusion[i,i] / confusion[:,i].sum()
# print(precision)
# # recall
# recall = torch.zeros(num_classes)
# for i in range(num_classes):
#     recall[i] = confusion[i,i] / confusion[i,:].sum()
# print(recall)
# # f1
# f1 = 2 * (precision * recall) / (precision + recall)
# print(f1)
# # macro
# macro = f1.mean()
# print(f"Macro: {macro:.4f}")