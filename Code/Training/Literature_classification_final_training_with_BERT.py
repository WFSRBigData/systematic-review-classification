# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:30:11 2020

@author: bulk007

Code to train a final BERT neural network model on all available training
data, based on the best parameters found during crossvalidation, to classify
papers in a specific topic as "Relevant" or "Not relevant" based on their
title and abstract. The used literature data has been selected from
academic databases based on keywords as labeled by human experts.
"""

# Import needed packages
import os
import re
import numpy as np
import tensorflow as tf
import ktrain
from ktrain import text as ktrain_text
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsRegressor

# Set random seed to be able to compare across models
np.random.seed(1)

# Set general parameters
case_study = 'Cereals' # Choose 'Cereals' or 'Leafy Greens'
model_name = 'BERT'
smote = False  # Set this to True in case of a big class imbalance
data_augmentation = False  # Set this to True in case there is little data available
data_path = os.path.join(os.getcwd(), '../..', 'Data', case_study)
save_path = os.path.join(os.getcwd(), '../..', 'Models', case_study)

# The used literature was supplied as Endnote library and was exported as a
# .txt file in BibTeX style per label (i.e. Relevant and Not Relevant). We
# load the .txt file, parse them and transform into a dataset consisting of the
# title+abstract as input and the label as output
with open(os.path.join(data_path, case_study.replace(" ", "") + '_Relevant.txt'),
          'r', encoding="utf8") as file:
    papers_relevant = file.read()
    papers_relevant = re.split(r'\@.*?{', papers_relevant)[1:]

with open(os.path.join(data_path, case_study.replace(" ", "") + '_NotRelevant.txt'),
          'r', encoding="utf8") as file:
    papers_not_relevant = file.read()
    papers_not_relevant = re.split(r'\@.*?{', papers_not_relevant)[1:]

text = []
labels = []

for entry in papers_relevant:
    if("title = {" in entry):
        labels.append(1)
        paper_text = entry.split("title = {")[1].split("},\n")[0] + ". "
        if("abstract = {" in entry):
            abstract = entry.split("abstract = {")[1].split("},\n")[0]
            paper_text = paper_text + abstract
        text.append(paper_text)

for entry in papers_not_relevant:
    if("title = {" in entry):
        labels.append(0)
        paper_text = entry.split("title = {")[1].split("},\n")[0] + ". "
        if("abstract = {" in entry):
            abstract = entry.split("abstract = {")[1].split("},\n")[0]
            paper_text = paper_text + abstract
        text.append(paper_text)

# Perform prepocessing on the text to remove special characters and numbers.
# After this remove the double spaces that might be added because of that step
text = [re.sub(r"[!\"’\'#$%&©±()×*+,-./:;<=>≥≤»→?@^_`{|}~\[\]\\0-9]", "", i) for i in text]
text = [re.sub(r"\s+", " ", i) for i in text]

# Shuffle dataset to get a good variation over the labels by first zipping
# the list together, then shuffle and unzip again
shuffle_zip = list(zip(text, labels))
np.random.shuffle(shuffle_zip)
text_shuffled, labels_shuffled = zip(*shuffle_zip)
text_shuffled = list(text_shuffled)
labels_shuffled = list(labels_shuffled)

# Split the data in 80% for the train and validation set and 20% for the test set
test_split = int(0.8*len(text_shuffled))
train_val_text = np.asarray(text_shuffled[0:test_split])
train_val_labels = np.asarray(labels_shuffled[0:test_split])

# A distilled model of BERT is used with less parameters as we do not have
# a lot of data. Preprocessing from text to numeric data is done in the
# code below in a way designed for the BERT algorithm.
tf.autograph.set_verbosity(0)
print("Preprocessing data")
bert_model = 'distilbert-base-uncased'
t = ktrain_text.Transformer(bert_model, maxlen=500, class_names=[0, 1])
train_val_preprocessed = t.preprocess_train(train_val_text, train_val_labels)

# In order to create a more balanced dataset SMOTE can be applied as
# oversampling technique. Depending on the imbalance between 'Relevant' and
# 'Not relevant' using SMOTE might be necessary
if(smote):
    print("Performing SMOTE")
    train_val_preprocessed_text = train_val_preprocessed.x.reshape(train_val_preprocessed.x.shape[0],
                                                    train_val_preprocessed.x.shape[1]*train_val_preprocessed.x.shape[2])
    train_val_preprocessed_labels = (train_val_preprocessed.y[:,1] == 1).astype(int)
    sm = SMOTE(random_state=42, k_neighbors=3)
    train_val_preprocessed_text, train_val_preprocessed_labels = sm.fit_sample(train_val_preprocessed_text, train_val_preprocessed_labels)
    train_val_preprocessed_text = train_val_preprocessed_text.reshape(train_val_preprocessed_text.shape[0],
                                                    train_val_preprocessed.x.shape[1], train_val_preprocessed.x.shape[2])
    train_val_preprocessed_labels = np.eye(2)[train_val_preprocessed_labels]
    train_val_preprocessed.x = train_val_preprocessed_text
    train_val_preprocessed.y = train_val_preprocessed_labels

# Use a nearest neighbors algorithm to augment the data to expand the train set by 100%
if(data_augmentation):
    print("Performing data augmentation")
    augmented_train_val_preprocessed = train_val_preprocessed.x.copy()
    augmented_train_labels = train_val_preprocessed.y.copy()   
    train_val_preprocessed_text = train_val_preprocessed.x.reshape(train_val_preprocessed.x.shape[0],
                                                    train_val_preprocessed.x.shape[1]*train_val_preprocessed.x.shape[2])
    train_val_preprocessed_labels = train_val_preprocessed.y
    knn = KNeighborsRegressor(4, 'distance').fit(train_val_preprocessed_text, train_val_preprocessed_labels)
    shuffled_indexes = list(range(len(train_val_preprocessed_text)))
    np.random.shuffle(shuffled_indexes)

    # Augment 20% of the train data and add it to the original set
    for index in shuffled_indexes[0:int(len(train_val_preprocessed_text)/5)]:
        datapoint_text = np.reshape(train_val_preprocessed_text[index], (1, -1))
        datapoint_label = train_val_preprocessed_labels[index].reshape(1, train_val_preprocessed_labels[index].shape[0])
        neighbor = knn.kneighbors(datapoint_text, return_distance=False)
        random_neighbor = np.random.randint(1, 4)
        difference = train_val_preprocessed_text[neighbor[0][random_neighbor]] - datapoint_text
        gap = np.random.rand(1)[0]
        new_point = (datapoint_text + difference*gap).astype(int)
        new_point = new_point.reshape(1,
                                      train_val_preprocessed.x.shape[1],
                                      train_val_preprocessed.x.shape[2])
        augmented_train_val_preprocessed = np.append(augmented_train_val_preprocessed, new_point, axis=0)
        augmented_train_labels = np.append(augmented_train_labels, datapoint_label, axis=0)

    train_val_preprocessed.x = augmented_train_val_preprocessed
    train_val_preprocessed.y = augmented_train_labels

# Initialize BERT classifier and fit it on the preprocessed training and
# data validation data
print("Training the model")
model = t.get_classifier()
learner = ktrain.get_learner(model, train_data=train_val_preprocessed, batch_size=2)
learner.fit_onecycle(5e-5, 3)

# Save the model
print("Saving the final model")
predictor = ktrain.get_predictor(learner.model, preproc=t)
if not os.path.exists(os.path.join(save_path, model_name, "Final")):
    os.makedirs(os.path.join(save_path, model_name, "Final"))
predictor.save(os.path.join(save_path, model_name, "Final", 'BERT_model'))
