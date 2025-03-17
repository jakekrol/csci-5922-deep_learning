#!/usr/bin/env python3
import string
import spacy
import json
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
from PIL import Image
from io import BytesIO
from torchvision import transforms

### data
data='./la3_data'
img_tr = os.path.join(data, 'train')
img_va = os.path.join(data, 'val')
img_te = os.path.join(data, 'test')
anno_tr = os.path.join(data, 'train.json')
anno_va = os.path.join(data, 'val.json')
anno_te = os.path.join(data, 'test.json')

### train annotations
with open(anno_tr) as f:
    # list of dicts
    a_tr = json.load(f)
print(len(a_tr))

### subset
train_subset = a_tr[:200]

### get top n answers to use as categories
# for multi-class classification instead of generation task
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
print(top_answers)

# Create categories for top n
category_name2id = {answer:ind for ind, (answer, _) in enumerate(top_answers)}
category_id2name = {ind:answer for ind, (answer, _) in enumerate(top_answers)}
category_id2name[top_n] = 'other_categories'

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

### one-hot encode vocab
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
