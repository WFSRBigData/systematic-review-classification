# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 16:27:41 2019

@author: bulk007

Code to train an LSTM neural network using 5-fold crossvalidation to classify
papers in a specific topic as "Relevant" or "Not relevant" based on their title
and abstract. The used literature data has been selected from academic databases
based on keywords as labeled by human experts.
"""

# Import needed packages
import os
import re
import math
import numpy as np
from numpy import interp
import silence_tensorflow.auto
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn.model_selection import KFold
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
result_path = os.path.join(os.getcwd(), '../..', 'Results', case_study, 'Crossvalidation')
nr_network_nodes = 12
learning_rate = 0.0005
lstm_dropout = 0.5
lstm_recurrent_dropout = 0
dropout = 0.5
l2_regularization = 0.0001
epochs = 50
batch_size = 32
num_tokens = 5000

# Create function to yield data during training so each batch can be balanced
# across classes
def generate_data(data, labels, batch_size=32, shuffle=True, 
                  balanced_batches=True):
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

# Define the amount of folds for the crossvalidation on the train and val set
kfold_cv = KFold(5)

# Apply 5-fold cross-validation so repeat steps 5 times
true_positive_rates = []
area_under_curves = []
fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
val_labels_list = []
val_predicted_prob_list = []
for i, (train_indices, val_indices) in enumerate(kfold_cv.split(train_val_text)):

    print("Fold {} of crossvalidation".format(i))

    # Split the train and validation set
    train_text = np.take(train_val_text, train_indices)
    train_labels = np.take(train_val_labels, train_indices)

    val_text = np.take(train_val_text, val_indices)
    val_labels = np.take(train_val_labels, val_indices)

    # We have to encode the words into numbers for the LSTM to work, so we will
    # train a tokenizer to encode the words. The words will be tokenized in
    # lowercase, and at the same time filtered from symbols, numbers and
    # punctuation as that does not contain any useful information
    print("Preprocess and tokenize data")
    to_filter = '!"’\'#$%&©±()×*+,-–−./:;<=>≥≤»→?@0123456789[\\]^_`{|}~\t\n'
    tokenizer = Tokenizer(num_words=num_tokens, filters=to_filter, lower=True)
    tokenizer.fit_on_texts(train_text)
    word_index = tokenizer.word_index
    nr_of_tokens = len(list(word_index.items()))+1  # Add 1 for the padd token
    
    # Apply the tokenizer to the train, validation and test set
    train_tokenized = tokenizer.texts_to_sequences(train_text)
    val_tokenized = tokenizer.texts_to_sequences(val_text)
    
    # Padd all the data with zeros to make sure all input is the same length
    max_length = max(len(tokenized_text) for tokenized_text in train_tokenized)
    train_data = pad_sequences(train_tokenized, maxlen=max_length,
                               padding='post', truncating='post', value=0)
    val_data = pad_sequences(val_tokenized, maxlen=max_length,
                             padding='post', truncating='post', value=0)
    
    # Transform to numpy array
    train_labels = np.asarray(list(train_labels))
    val_labels = np.asarray(list(val_labels))
    
    # In order to create a more balanced dataset SMOTE can be applied as
    # oversampling technique. Depending on the imbalance between 'Relevant' and
    # 'Not relevant' using SMOTE might be necessary
    if(smote):
        print("Performing SMOTE")
        sm = SMOTE(random_state=42, k_neighbors=3)
        train_data, train_labels = sm.fit_sample(train_data, train_labels)
    
    # Use a nearest neighbors algorithm to augment the data to expand the train set
    if(data_augmentation):
        print("Performing data augmentation")
        augmented_train_data = list(train_data.copy())
        augmented_train_labels = list(train_labels.copy())
        knn = KNeighborsRegressor(4, 'distance').fit(train_data, train_labels)
        shuffled_indexes = list(range(len(augmented_train_data)))
        np.random.shuffle(shuffled_indexes)
        
        # Augment 20% of the train data and add it to the original set
        for index in shuffled_indexes[0:int(len(augmented_train_data)/5)]:  
            datapoint_text = np.reshape(augmented_train_data[index], (1, -1))
            datapoint_label = augmented_train_labels[index]      
            neighbor = knn.kneighbors(datapoint_text, return_distance=False)
            random_neighbor = np.random.randint(1,4)
            difference = train_data[neighbor[0][random_neighbor]] - datapoint_text
            gap = np.random.rand(1)[0]
            new_point = datapoint_text + difference*gap
            augmented_train_data = np.append(augmented_train_data, new_point, axis=0)
            augmented_train_labels.append(datapoint_label)
    
        train_data = np.asarray(list(augmented_train_data))
        train_labels = np.asarray(list(augmented_train_labels))

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
    train_generator = generate_data(train_data, train_labels, batch_size=32,
                                    shuffle=True, balanced_batches=True)
    validation_generator = generate_data(val_data, val_labels, batch_size=32,
                                         shuffle=True, balanced_batches=True)

    # Set train parameters and train the model
    print("Nodes: {}, LR: {}, lstm_dropout: {}, lstm_rec_dropout: {}, dropout: {}, l2: {}".format(
            nr_network_nodes, learning_rate, lstm_dropout,
            lstm_recurrent_dropout, dropout, l2_regularization))
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=['accuracy'])
    model_save_name = "LSTM-{}-{}-{}-fold{}".format(nr_network_nodes,
                                                       learning_rate,
                                                       num_tokens, i) + "-{epoch:02d}.hdf5"
    progress_history = model.fit_generator(train_generator,
                                           steps_per_epoch=math.ceil(len(train_data)/batch_size),
                                           epochs=epochs, verbose=2,
                                           validation_data=validation_generator,
                                           validation_steps=math.ceil(len(val_data)/batch_size),
                                           workers=0)

    # Save the model and tokenizer for this fold
    print("Saving model and testing on validation set")
    if not os.path.exists(os.path.join(save_path, model_name, "Fold_{}".format(i))):
        os.makedirs(os.path.join(save_path, model_name, "Fold_{}".format(i)))
    pickle.dump(tokenizer, open(os.path.join(save_path, model_name, "Fold_{}".format(i), model_name+'_tokenizer.pkl'), 'wb'))
    model.save(os.path.join(save_path, model_name, "Fold_{}".format(i), 'LSTM_model.h5'))

    # Predict on the validation data
    val_prob_predicted = model.predict(val_data).ravel()
    val_labels_list = val_labels_list + list(val_labels)
    val_predicted_prob_list = val_predicted_prob_list + list(val_prob_predicted)

    # Plot the ROC curve and save the true positive rate and the area under the curve
    false_positive_rate, true_positive_rate, thresholds = roc_curve(val_labels, val_prob_predicted)
    auc_value = auc(false_positive_rate, true_positive_rate)
    ax.plot(false_positive_rate, true_positive_rate,
            label='ROC fold {} (AUC = {:.2f})'.format(i+1, auc_value),
            alpha=0.3, lw=1)

    # To calculate the mean over the folds, we need a consistent number of
    # thresholds per fold, which we force by interpolation over a standard
    # length of 100 values
    mean_false_positive_rates = np.linspace(0, 1, 100)
    true_positive_rate_interp = interp(mean_false_positive_rates, false_positive_rate, true_positive_rate)
    true_positive_rate_interp[0] = 0  # Set the initial value to 0 as it should start there
    true_positive_rate_interp[-1] = 1  # Set the last value to 1 as it should end there
    true_positive_rates.append(true_positive_rate_interp)
    area_under_curves.append(auc_value)

    # Output graphs to show performance: loss, accuracy and AU
    fig_loss, ax_loss = plt.subplots()
    ax_loss.plot(progress_history.history['loss'])
    ax_loss.plot(progress_history.history['val_loss'], '')
    ax_loss.set_xlabel("Epochs")
    ax_loss.set_ylabel('Loss')
    ax_loss.legend(['loss', 'validation loss'], loc='upper right')
    fig_loss.show()

    fig_acc, ax_acc = plt.subplots()
    ax_acc.plot(progress_history.history['accuracy'])
    ax_acc.plot(progress_history.history['val_accuracy'], '')
    ax_acc.set_xlabel("Epochs")
    ax_acc.set_ylabel('Accuracy')
    ax_acc.legend(['accuracy', 'validation accuracy'], loc='lower right')
    fig_acc.show()

    lowest_val_loss = min(progress_history.history['val_loss'])
    corresponding_accuracy = progress_history.history['val_accuracy'][np.argmin(progress_history.history['val_loss'])]
    print("Lowest validation loss: {}, with corresponding accuracy: {}".format(
            lowest_val_loss, corresponding_accuracy))

    tf.keras.backend.clear_session() # Clear session to prevent memory leak from TF

# Calculate the averages over the cross validation
mean_true_positive_rates = np.mean(true_positive_rates, axis=0)
mean_area_under_curve = auc(mean_false_positive_rates, mean_true_positive_rates)
std_area_under_curve = np.std(area_under_curves)
std_true_positive_rates = np.std(true_positive_rates, axis=0)
upper_bound_roc = np.minimum(mean_true_positive_rates + std_true_positive_rates, 1)
lower_bound_roc = np.maximum(mean_true_positive_rates - std_true_positive_rates, 0)

# Check if results folder exists for this case
if not os.path.exists(result_path):
    os.makedirs(result_path)

# Plot the ROC curves and save to disk
ax.plot(mean_false_positive_rates, mean_true_positive_rates, color='b',
        label='Mean ROC (AUC = {:.2f} $\pm$ {:.2f})'.format(
                mean_area_under_curve, std_area_under_curve),
        lw=2, alpha=.8)
ax.fill_between(mean_false_positive_rates, lower_bound_roc, upper_bound_roc,
                color='grey', alpha=.2, label=r'$\pm$ 1 std. dev. from Mean ROC')
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="ROC curve")
ax.legend(loc="lower right")
fig.savefig(os.path.join(result_path, "5-fold_crossvalidation_ROC_{}.png".format(model_name)))
fig.show()

# Determine the final predictions for this model
val_predicted_list = np.where(val_predicted_prob_list < np.float64(0.5), 0, 1)

# Save the final results in a .txt file
results_file = open(os.path.join(result_path, "5-fold_crossvalidation_results_{}.txt".format(model_name)), 
                    'w', encoding="utf8")

print("\nSMOTE = {} and augmentation = {}\n".format(smote, data_augmentation))
results_file.write("SMOTE = {} and augmentation = {}\n".format(smote, data_augmentation))

# Print the classification report containing precision, recall and F1-score
report = classification_report(val_labels_list, val_predicted_list, digits=3)
print(report)
results_file.write(report + "\n\n")

# Print the confusion matrix
confusion = confusion_matrix(val_labels_list, val_predicted_list)
error = confusion[0, 1] + confusion[1, 0]
print(confusion)
print("\nNumber of errors: {}".format(error))
results_file.write(str(confusion))
results_file.write("\n\nNumber of errors: {}".format(error))

# Close the .txt file
results_file.close()
