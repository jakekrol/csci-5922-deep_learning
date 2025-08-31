#!/usr/bin/env python3
import string
import spacy
import json
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import Counter
from PIL import Image
from io import BytesIO
from torchvision import transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### data
data='./la3_data'
img_tr = os.path.join(data, 'train')
img_va = os.path.join(data, 'val')
img_te = os.path.join(data, 'test')
anno_tr = os.path.join(data, 'train.json')
anno_va = os.path.join(data, 'val.json')
anno_te = os.path.join(data, 'test.json')

### annotations
with open(anno_tr,'r') as f:
    # list of dicts
    a_tr = json.load(f)
with open(anno_va,'r') as f:
    a_val = json.load(f)
with open(anno_te,'r') as f:
    a_te = json.load(f)

### subset
train_subset = a_tr[:200] # set to 1000

##### train data processing
### get top answers for challenge 2 labels
chosen_answers = []
top_n = 100
# Choose the most common answer from the ten labels
for sample in train_subset:
    answers = [entry['answer'] for entry in sample['answers']]
    answer_counts = Counter(answers) # set up counter
    top_answer, _ = answer_counts.most_common(1)[0] # get top answer out of the ten
    chosen_answers.append(top_answer)
# Count the frequency of answers for all samples
answer_counts = Counter(chosen_answers)
top_answers = answer_counts.most_common(top_n) # top n
# maps labels to indices
category_name2id = {answer:ind for ind, (answer, _) in enumerate(top_answers)}
category_id2name = {ind:answer for ind, (answer, _) in enumerate(top_answers)}
category_id2name[top_n] = 'other_categories'
# write labels and indices
with open('challenge2_name2id.json', 'w') as f:
    json.dump(category_name2id, f)
### train challenge 2 targets
targets_trc2 = []
for ans in chosen_answers:
    if ans in category_name2id.keys():
        targets_trc2.append(category_name2id[ans])
    else:
        targets_trc2.append(top_n) # if not in top n, use the n+1 th category

targets_t_trc2 = torch.tensor(targets_trc2)
print('Target size:', targets_t_trc2.shape)

### transform images
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize (you can adjust the size)
    transforms.ToTensor(),
])
image_tensors = []
for sample in train_subset:
    image_url = os.path.join(img_tr, sample['image'])
    img = Image.open(image_url)
    img_tensor = transform(img)
    image_tensors.append(img_tensor)
image_tensors = torch.stack(image_tensors)
print('Train images shape', image_tensors.shape)
# one-hot encode vocab
questions = [sample['question'] for sample in train_subset]
nlp = spacy.load('en_core_web_sm')
tokenized_questions = []
# Tokenize, remove stop words, lemmatize, and remove punctuation
for question in questions:
    doc = nlp(question.lower())  # lowercase
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    tokenized_questions.append(tokens)
# Create the vocabulary (unique words from all questions)
vocabulary = set(word for question in tokenized_questions for word in question)
vocabulary = sorted(vocabulary)
vocab_size = len(vocabulary)
print('Vocabulary size', len(vocabulary))
# Generate the one-hot encoding manually
bag_of_words = []
for question in tokenized_questions:
    BoW_vector = torch.zeros(vocab_size)
    for ind, voc in enumerate(vocabulary):
        if voc in question:
            BoW_vector[ind] = 1
    bag_of_words.append(BoW_vector)

question_tensors = torch.stack(bag_of_words)
print('Train questions shape', question_tensors.shape)

### process validation
val_subset = a_val[:100]
# process question text
val_questions = [sample['question'] for sample in val_subset]
val_tokenized_questions = []
# Tokenize, remove stop words, lemmatize, and remove punctuation
for question in val_questions:
    doc = nlp(question.lower())  # lowercase
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    val_tokenized_questions.append(tokens)
# Generate the one-hot encoding manually
bag_of_words = []
for question in val_questions:
    BoW_vector = torch.zeros(vocab_size)
    for ind, voc in enumerate(vocabulary):
        if voc in question:
            BoW_vector[ind] = 1
    bag_of_words.append(BoW_vector)
val_question_tensors = torch.stack(bag_of_words)
print('Validation questions shape', val_question_tensors.shape)
# process validation images
val_image_tensors = []
for sample in val_subset:
    image_url = os.path.join(img_va, sample['image'])
    img = Image.open(image_url)
    img_tensor = transform(img)
    val_image_tensors.append(img_tensor)
val_image_tensors = torch.stack(val_image_tensors)
print('validation images shape', val_image_tensors.shape)
# choose top answer
vchosen_answers=[]
for sample in val_subset:
    answers = [entry['answer'] for entry in sample['answers']]
    answer_counts = Counter(answers) # set up counter
    top_answer, _ = answer_counts.most_common(1)[0] # get top answer out of the ten
    vchosen_answers.append(top_answer)
### validation targets challenge 2
targets_vac2 = []
for ans in vchosen_answers:
    if ans in category_name2id.keys():
        targets_vac2.append(category_name2id[ans])
    else:
        targets_vac2.append(top_n) # if not in top n, use the n+1 th category

targets_t_vac2 = torch.tensor(targets_vac2)
print('Validation target size:', targets_t_vac2.shape)

##### challenge 1 targets
def weigh_class(tensor):
    #w_j = n/n_j * k
    n = len(tensor)
    d = dict(Counter(tensor))
    k=len(d.keys())
    w = {}
    for j in range(k):
        n_j = d[j]
        w[j] = n/n_j*k
    return w
### train
targets_trc1 = []
for sample in train_subset:
    targets_trc1.append(sample['answerable'])
print('Challenge 1 train target size:', len(targets_trc1))
print(targets_trc1[:10])
targets_t_trc1 = torch.tensor(targets_trc1)
print("Class balance of training set in challenge 1:", Counter(targets_trc1))
target_weights_c1 = weigh_class(targets_trc1)

### val
targets_vac1 = []
for sample in val_subset:
    targets_vac1.append(sample['answerable'])
print('Challenge 1 val target size:', len(targets_vac1))
print(targets_vac1[:10])
targets_t_vac1 = torch.tensor(targets_vac1)
print('Class balance of validation set in challenge 1:', Counter(targets_vac1))

##### process test data
test = a_te[:100] #! only submit the first 100
# process question text
te_questions = [sample['question'] for sample in test]
te_tokenized_questions = []
# Tokenize, remove stop words, lemmatize, and remove punctuation
for question in te_questions:
    doc = nlp(question.lower())  # lowercase
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    te_tokenized_questions.append(tokens)
# Generate the one-hot encoding manually
bag_of_words = []
for question in te_questions:
    BoW_vector = torch.zeros(vocab_size)
    for ind, voc in enumerate(vocabulary):
        if voc in question:
            BoW_vector[ind] = 1
    bag_of_words.append(BoW_vector)
te_question_tensors = torch.stack(bag_of_words)
print('Test questions shape', te_question_tensors.shape)
# process test images
te_image_tensors = []
for sample in test:
    image_url = os.path.join(img_te, sample['image'])
    img = Image.open(image_url)
    img_tensor = transform(img)
    te_image_tensors.append(img_tensor)
te_image_tensors = torch.stack(te_image_tensors)
print('Test images shape', te_image_tensors.shape)

# challenge 1
class VQAModelC1(nn.Module):
    def __init__(self, vocab_size, output_classes, image_channels=3, image_size=128, hidden_dim=512):
        super(VQAModelC1, self).__init__()

        # Image processing part
        self.conv1 = nn.Conv2d(in_channels=image_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers after convolutional layers
        self.fc_image = nn.Linear(128 * (image_size // 8) * (image_size // 8), hidden_dim)

        # Question processing part (fully connected)
        self.fc_question1 = nn.Linear(vocab_size, 512)
        self.fc_question2 = nn.Linear(512, hidden_dim)

        # Fusion part: Multiply image and question features
        self.fc_fusion = nn.Linear(hidden_dim, hidden_dim)

        # Final classification layer
        self.fc_output = nn.Linear(hidden_dim, output_classes)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, image, question):
        # Image part
        x_image = F.relu(self.conv1(image))
        x_image = self.pool1(x_image)
        x_image = F.relu(self.conv2(x_image))
        x_image = self.pool2(x_image)
        x_image = F.relu(self.conv3(x_image))
        x_image = self.pool3(x_image)
        x_image = x_image.view(x_image.size(0), -1) # flatten
        x_image = F.relu(self.fc_image(x_image))

        # Question part (Fully Connected)
        x_question = F.relu(self.fc_question1(question))
        x_question = F.relu(self.fc_question2(x_question))

        # Fusion: Multiply image and question features
        x_fusion = x_image * x_question

        # Apply fusion layer
        x_fusion = F.relu(self.fc_fusion(x_fusion))

        # Output classification
        output = self.fc_output(x_fusion)
        return output

# challenge 2
class VQAModelC2(nn.Module):
    def __init__(self, vocab_size, output_classes, image_channels=3, image_size=128, hidden_dim=512):
        super(VQAModelC2, self).__init__()

        # Image processing part
        self.conv1 = nn.Conv2d(in_channels=image_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers after convolutional layers
        self.fc_image = nn.Linear(128 * (image_size // 8) * (image_size // 8), hidden_dim)

        # Question processing part (fully connected)
        self.fc_question1 = nn.Linear(vocab_size, 512)
        self.fc_question2 = nn.Linear(512, hidden_dim)

        # Fusion part: Multiply image and question features
        self.fc_fusion = nn.Linear(hidden_dim, hidden_dim)

        # Final classification layer
        self.fc_output = nn.Linear(hidden_dim, output_classes)

    def forward(self, image, question):
        # Image part
        x_image = F.relu(self.conv1(image))
        x_image = self.pool1(x_image)
        x_image = F.relu(self.conv2(x_image))
        x_image = self.pool2(x_image)
        x_image = F.relu(self.conv3(x_image))
        x_image = self.pool3(x_image)
        x_image = x_image.view(x_image.size(0), -1) # flatten
        x_image = F.relu(self.fc_image(x_image))

        # Question part (Fully Connected)
        x_question = F.relu(self.fc_question1(question))
        x_question = F.relu(self.fc_question2(x_question))

        # Fusion: Multiply image and question features
        x_fusion = x_image * x_question

        # Apply fusion layer
        x_fusion = F.relu(self.fc_fusion(x_fusion))

        # Output classification
        output = self.fc_output(x_fusion)
        return output
##### challenge 1 train, validation and test
print("Challenge 1")
# model
output_classes = 1
model = VQAModelC1(vocab_size, output_classes).to(device)
criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Training loop
num_epochs = 30
# num_epochs = 5
acctrain=[]
accval=[]
for epoch in range(num_epochs):
    model.train()
    images = image_tensors.to(device)
    questions = question_tensors.to(device)
    targets = targets_t_trc1.to(device)
    # forward
    optimizer.zero_grad()
    outputs = model(images, questions).squeeze()
    loss = criterion(outputs, targets.float())
    probabilities = torch.sigmoid(outputs)
    # do weighted bce
    # loss = -torch.mean((target_weights_c1[1] * (targets * torch.log(probabilities))) + (target_weights_c1[0]*(1 - targets) * torch.log(1 - probabilities)))
    # loss = -torch.mean((target_weights_c1[1] * (targets * torch.log(probabilities))) + (target_weights_c1[0] * ((1 - targets) * torch.log(1 - probabilities))))
    # loss = criterion(outputs, targets.float())
    # backward
    loss.backward()
    optimizer.step()
    # accuracy
    predicted = (probabilities > 0.5).float()
    correct = (predicted == targets).sum().item()  # Count correct predictions
    accuracy = correct / targets.size(0)  # Calculate accuracy
    acctrain.append(accuracy)

    # Print statistics
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()/len(targets):.4f}, Accuracy: {accuracy:.4f}")
        # Validation loop
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():  # Disable gradient computation for validation
        images = val_image_tensors.to(device)
        questions = val_question_tensors.to(device)
        targets = targets_t_vac1.to(device)
        outputs = model(images, questions)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == targets).sum().item()
        accuracy = correct / targets.size(0)
        print(f"Validation Accuracy: {accuracy:.4f}")
        accval.append(accuracy)

# plot acc
plt.plot(acctrain, label='train')
plt.plot(accval, label='val')
plt.legend()
plt.savefig('c1_acc.png')

# save challenge1 results
model.eval()  # Set model to evaluation mode
with torch.no_grad():  # Disable gradient computation for validation
    images = te_image_tensors.to(device)
    questions = te_question_tensors.to(device)
    outputs = model(images, questions)
    _, predicted = torch.max(outputs, 1)

predictions = predicted.cpu().numpy()
torch.save(predictions,'jacob_krol_challenge1.pkl')
sys.exit()

##### challenge 2 train, validation and test
print("Challenge 2")
# There will be top_n classes plus "others"
output_classes = top_n + 1

model = VQAModelC2(vocab_size, output_classes).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
# num_epochs = 30
num_epochs = 10
acctrain=[]
accval=[]
for epoch in range(num_epochs):
    model.train()
    # to device
    images = image_tensors.to(device)
    questions = question_tensors.to(device)
    targets = targets_t_trc2.to(device)

    # forward
    optimizer.zero_grad()
    outputs = model(images, questions)
    # backward
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    _, predicted = torch.max(outputs, 1)  # Set predicted category
    correct = (predicted == targets).sum().item()  # Count correct predictions
    accuracy = correct / targets.size(0)  # Calculate accuracy
    # append acc
    acctrain.append(accuracy)

    # Print statistics
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()/len(targets):.4f}, Accuracy: {accuracy:.4f}")
        # Validation loop
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():  # Disable gradient computation for validation
        images = val_image_tensors.to(device)
        questions = val_question_tensors.to(device)
        targets = targets_t_vac2.to(device)
        outputs = model(images, questions)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == targets).sum().item()
        accuracy = correct / targets.size(0)
        print(f"Validation Accuracy: {accuracy:.4f}")
        accval.append(accuracy)

# plot acc
plt.plot(acctrain, label='train')
plt.plot(accval, label='val')
plt.legend()
plt.savefig('c2_acc.png')

        
### inference on test set

### challenge 2
model.eval()  # Set model to evaluation mode

with torch.no_grad():  # Disable gradient computation for validation
    images = te_image_tensors.to(device)
    questions = te_question_tensors.to(device)
    outputs = model(images, questions)
    _, predicted = torch.max(outputs, 1)

predictions = predicted.cpu().numpy()
image_names = [sample['image'] for sample in test]
answers = [category_id2name[pred] for pred in predictions]
results = [{'image':image, 'answer':ans} for image, ans in zip(image_names, answers)]

print('Check the format for the first sample:', results[0])

# write to json
with open('jacob_krol_challenge2.json', 'w') as f:
    json.dump(results, f)