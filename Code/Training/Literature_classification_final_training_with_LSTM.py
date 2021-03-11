# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 16:13:06 2020

@author: bulk007

Code to train a final LSTM neural network model on all available training data
to classify papers in a specific topic as "Relevant" or "Not relevant" based on
their title and abstract. The used literature data has been selected from
academic databases based on keywords as labeled by human experts.
"""


# Import needed packages
import os
import re
import math
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsRegressor
import tensorflow as tf
import pickle

# Set random seed to be able to compare across models
np.random.seed(1)

# Set general parameters
case_study = 'Cereals' # Choose 'Cereals' or 'Leafy Greens'
model_name = 'LSTM'
smote = False  # Set this to True in case of a big class imbalance
data_augmentation = False  # Set this to True in case there is little data available
data_path = os.path.join(os.getcwd(), '../..', 'Data', case_study)
save_path = os.path.join(os.getcwd(), '../..', 'Models', case_study)
nr_network_nodes = 12
learning_rate = 0.0005
lstm_dropout = 0.5
lstm_recurrent_dropout = 0
dropout = 0.5
l2_regularization = 0.0001
epochs = 50
batch_size = 32
num_tokens = 5000
max_augmentation = 50

# Create function to yield data during training so data augmentation can happen on the fly
def generate_data(data, labels, batch_size=32, shuffle=True,
                  balanced_batches=True, augmentation=False, num_tokens=0):
    """
    Generator to make batches to train a model on data while having the option
    to augment and balance the data.
    # Params
    - data : Array containing the data to generate from
    - labels : Array containg the labels for the given data
    - batch_size : number of data points per batch
    - shuffle : boolean to shuffle data in a random order
    - balanced_batches : If true, balances batches so each class is
                        equally represented
    # Returns
    - batch of data (batch_size, data.shape[1])
    - batch of labels (batch_size,)
    """

    while True:
        data_length = len(data)

        # Shuffle data if shuffle=True
        if shuffle:
            shuffle_zip = list(zip(data, labels))
            np.random.shuffle(shuffle_zip)
            data, labels = zip(*shuffle_zip)

        # Dubble data and labels in case data is not well devided by the batch
        # size and we need to spill over to fill the last batch
        data = np.concatenate((data, data))
        labels = np.concatenate((labels, labels))

        # Loop over all the data and create batches
        data_nr = 0
        while data_nr < data_length:
            data_batch = np.zeros((batch_size, data.shape[1]), dtype=np.int32)
            label_batch = np.zeros((batch_size,), dtype=np.int32)
            data_nr_batch = 0
            label_count = np.zeros(2)  # create a counter to track the balancing of batches

            while data_nr_batch < batch_size:
                try:
                    data_point = data[data_nr, :]
                except:
                    print(data.size, data_nr)
                    break
                try:
                    label = labels[data_nr]
                except:
                    print(labels.size)
                    break
                data_nr = data_nr + 1

                if(balanced_batches and label_count[label] >= batch_size/2):
                    continue  # continue if one label has enough occurences

                label_count[label] = label_count[label]+1

                data_batch[data_nr_batch] = data_point
                label_batch[data_nr_batch] = label
                data_nr_batch = data_nr_batch+1
            
            yield data_batch, label_batch

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

# We have to encode the words into numbers for the LSTM to work, so we will
# train a tokenizer to encode the words. The words will be tokenized in
# lowercase, and at the same time filtered from symbols, numbers and
# punctuation as that does not contain any useful information
print("Preprocess and tokenize data")
to_filter = '!"’\'#$%&©±()×*+,-–−./:;<=>≥≤»→?@0123456789[\\]^_`{|}~\t\n'
tokenizer = Tokenizer(num_words=num_tokens, filters=to_filter, lower=True)
tokenizer.fit_on_texts(train_val_text)
word_index = tokenizer.word_index
nr_of_tokens = len(list(word_index.items()))+1  # Add 1 for the padd token

# Apply the tokenizer to the train, validation and test set
train_val_tokenized = tokenizer.texts_to_sequences(train_val_text)

# Padd all the data with zeros to make sure all input is the same length
max_length = max(len(tokenized_text) for tokenized_text in train_val_tokenized)
train_val_data = pad_sequences(train_val_tokenized, maxlen=max_length,
                           padding='post', truncating='post', value=0)

# Transform labels to numpy array to match type of input data
train_val_labels = np.asarray(list(train_val_labels))

# In order to create a more balanced dataset SMOTE can be applied as
# oversampling technique. Depending on the imbalance between 'Relevant' and
# 'Not relevant' using SMOTE might be necessary
if(smote):
    print("Performing SMOTE")
    sm = SMOTE(random_state=42, k_neighbors=3)
    train_val_data, train_val_labels = sm.fit_sample(train_val_data, train_val_labels)

# Use a nearest neighbors algorithm to augment the data to expand the train set
if(data_augmentation):
    print("Performing data augmentation")
    augmented_train_val_data = list(train_val_data.copy())
    augmented_train_val_labels = list(train_val_labels.copy())
    knn = KNeighborsRegressor(4, 'distance').fit(train_val_data, train_val_labels)
    shuffled_indexes = list(range(len(augmented_train_val_data)))
    np.random.shuffle(shuffled_indexes)
    
    # Augment 20% of the train data and add it to the original set
    for index in shuffled_indexes[0:int(len(augmented_train_val_data)/5)]:  
        datapoint_text = np.reshape(augmented_train_val_data[index], (1, -1))
        datapoint_label = augmented_train_val_labels[index]      
        neighbor = knn.kneighbors(datapoint_text, return_distance=False)
        random_neighbor = np.random.randint(1,4)
        difference = train_val_data[neighbor[0][random_neighbor]] - datapoint_text
        gap = np.random.rand(1)[0]
        new_point = datapoint_text + difference*gap
        augmented_train_val_data = np.append(augmented_train_val_data, new_point, axis=0)
        augmented_train_val_labels.append(datapoint_label)

    train_val_data = np.asarray(list(augmented_train_val_data))
    train_val_labels = np.asarray(list(augmented_train_val_labels))

# Create the LSTM network
tf.autograph.set_verbosity(0)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(num_tokens, nr_network_nodes),
    tf.keras.layers.Dropout(dropout),
    tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(nr_network_nodes, dropout=lstm_dropout,
                                 recurrent_dropout=lstm_recurrent_dropout)),
    tf.keras.layers.Dropout(dropout),
    tf.keras.layers.Dense(nr_network_nodes,
                          kernel_regularizer=tf.keras.regularizers.l2(l2_regularization),
                          activation='relu'),
    tf.keras.layers.Dropout(dropout),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
print(model.summary())

# Create train and validation generators
train_generator = generate_data(train_val_data, train_val_labels, batch_size=32,
                                shuffle=True, balanced_batches=True,
                                augmentation=True, num_tokens=num_tokens)

# Set train parameters and train the model
print("Nodes: {}, LR: {}, lstm_dropout: {}, lstm_rec_dropout: {}, dropout: {}, l2: {}".format(
        nr_network_nodes, learning_rate, lstm_dropout,
        lstm_recurrent_dropout, dropout, l2_regularization))
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              metrics=['accuracy'])
model_save_name = "LSTM-{}-{}-{}-{}-final.hdf5".format(nr_network_nodes, 
                                                     learning_rate, 
                                                     num_tokens,
                                                     max_augmentation)
progress_history = model.fit_generator(train_generator,
                                       steps_per_epoch=math.ceil(len(train_val_data)/batch_size),
                                       epochs=epochs, verbose=2)

# Save the model and tokenizer
print("Saving model and testing on test set")
if not os.path.exists(os.path.join(save_path, model_name, "Final")):
    os.makedirs(os.path.join(save_path, model_name, "Final"))
pickle.dump(tokenizer, open(os.path.join(save_path, model_name, "Final", model_name+'_tokenizer.pkl'), 'wb'))
model.save(os.path.join(save_path, model_name, "Final", 'LSTM_model.h5'))
