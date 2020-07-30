# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 16:21:08 2020

@author: bulk007

Code to train a final Adaboost model on all available training
data, based on the best parameters found during crossvalidation,
to classify papers in a specific topic as "Relevant" or "Not relevant"
based on their title and abstract. The used literature data has been selected
from academic databases based on keywords as labeled by human experts.
"""

# Import needed packages
import os
import re
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostClassifier
import pickle

# Set random seed to be able to compare across models
np.random.seed(1)

# Set general parameters
case_study = 'Cereals' # Choose 'Cereals' or 'Leafy Greens'
model_name = 'AdaBoost'
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
train_val_text = text_shuffled[0:test_split]
train_val_labels = labels_shuffled[0:test_split]

# We have to transform our data into numeric vectors for Logistic Regression
# to work, we will first create a document-term matrix for the training set
print("Vectorizing data and applying TFIDF")
count_vectorizer = CountVectorizer(lowercase=True, stop_words='english',
                                   ngram_range=(1, 1), tokenizer=None)
trained_vectorizer = count_vectorizer.fit(train_val_text)
text_tf = trained_vectorizer.transform(train_val_text)

# We will divide the counts by frequency to not favour longer texts
tfidf_transformer = TfidfTransformer()
trained_transformer = tfidf_transformer.fit(text_tf)
text_tfidf = trained_transformer.transform(text_tf)

# In order to create a more balanced dataset SMOTE can be applied as
# oversampling technique. Depending on the imbalance between 'Relevant' and
# 'Not relevant' using SMOTE might be necessary
if(smote):
    print("Performing SMOTE")
    sm = SMOTE(random_state=42, k_neighbors=3)
    text_tfidf, train_val_labels = sm.fit_sample(text_tfidf, train_val_labels)

# Use a nearest neighbors algorithm to augment the data to expand the train set
if(data_augmentation):
    print("Performing data augmentation")
    text_tfidf = text_tfidf.toarray()
    augmented_train_tfidf = list(text_tfidf.copy())
    augmented_train_labels = list(train_val_labels.copy())
    knn = KNeighborsRegressor(4, 'distance').fit(text_tfidf, train_val_labels)
    shuffled_indexes = list(range(len(augmented_train_tfidf)))
    np.random.shuffle(shuffled_indexes)
    
    # Augment 20% of the train data and add it to the original set
    for index in shuffled_indexes[0:int(len(augmented_train_tfidf)/5)]:  
        datapoint_text = np.reshape(augmented_train_tfidf[index], (1, -1))
        datapoint_label = augmented_train_labels[index]      
        neighbor = knn.kneighbors(datapoint_text, return_distance=False)
        random_neighbor = np.random.randint(1,4)
        difference = text_tfidf[neighbor[0][random_neighbor]] - datapoint_text
        gap = np.random.rand(1)[0]
        new_point = datapoint_text + difference*gap
        augmented_train_tfidf = np.append(augmented_train_tfidf, new_point, axis=0)
        augmented_train_labels.append(datapoint_label)

    text_tfidf = sparse.csr_matrix(augmented_train_tfidf)
    train_val_labels = augmented_train_labels

# Initialize AdaBoost classifier and fit it on the tf-idf training data
print("Training the model")
classifier = AdaBoostClassifier(n_estimators=1000)
trained_classifier = classifier.fit(text_tfidf, train_val_labels)

# Save the vectorizer, transformer and model for this fold
print("Saving final model")
model_name = model_name.replace(" ","")
if not os.path.exists(os.path.join(save_path, model_name, "Final")):
    os.makedirs(os.path.join(save_path, model_name, "Final"))
pickle.dump(trained_vectorizer, open(os.path.join(save_path, model_name, "Final",
                                                  model_name+'_vectorizer.pkl'), 'wb'))
pickle.dump(trained_transformer, open(os.path.join(save_path, model_name, "Final",
                                                   model_name+'_transformer.pkl'), 'wb'))
pickle.dump(trained_classifier, open(os.path.join(save_path, model_name, "Final",
                                                  model_name+'_classifier.pkl'), 'wb'))
