# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 11:54:28 2020

@author: bulk007

Code to test a specified ensemble of trained models to classify papers on an
external test set in a specific topic as "Relevant" or "Not relevant" based on
their title and abstract. The used literature data has been selected from
academic databases based on keywords as labeled by human experts.
"""

# Import needed packages
import os
import re
import numpy as np
from numpy import interp
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import ktrain
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pickle

# Set random seed to be able to compare across models
np.random.seed(1)

# Set general parameters
case_study = 'Cereals' # Choose 'Cereals' or 'Leafy Greens'
model_name = 'Ensemble'
data_path = os.path.join(os.getcwd(), '../../..', 'Data', case_study)
model_path = os.path.join(os.getcwd(), '../../..', 'Models', case_study)
result_path = os.path.join(os.getcwd(), '../../..', 'Results', case_study, 'Ensemble')
models = ['SVM','NaiveBayes'] # Add the name(s) of the models you want to use in a list separated with comma's
threshold = 0.25 # Set the threshold you want to use

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
test_text = text_shuffled[test_split:]
test_labels = labels_shuffled[test_split:]

# Loop over the selected models, load the correct files and make predictions
# on the test set
model_predicted_prob_list = []
for model in models:

    print("Testing with " + model)

    if(model == 'LSTM'):
        tf.autograph.set_verbosity(0)
        tokenizer = pickle.load(open(os.path.join(model_path, model, "Final", model+'_tokenizer.pkl'), 'rb'))
        classifier = tf.keras.models.load_model(os.path.join(model_path, model, "Final", 'LSTM_model.h5'))

        test_tokenized = tokenizer.texts_to_sequences(test_text)
        test_data = pad_sequences(test_tokenized, maxlen=543,  # The maxlen was determined when training the original model
                                  padding='post', truncating='post',
                                  value=0)
        model_predicted_prob = classifier.predict(test_data).ravel()
        model_predicted_prob_list.append(model_predicted_prob)
        tf.keras.backend.clear_session() # Clear session to prevent memory leak from TF

    elif(model == 'BERT'):
        tf.autograph.set_verbosity(0)
        classifier = ktrain.load_predictor(os.path.join(model_path, model, "Final", 'BERT_model'))
        model_predicted_prob = classifier.predict_proba(test_text)[:, 1]
        model_predicted_prob_list.append(model_predicted_prob)
        tf.keras.backend.clear_session() # Clear session to prevent memory leak from TF

    else:
        vectorizer = pickle.load(open(os.path.join(model_path, model, "Final", model+'_vectorizer.pkl'), 'rb'))
        transformer = pickle.load(open(os.path.join(model_path, model, "Final", model+'_transformer.pkl'), 'rb'))
        classifier = pickle.load(open(os.path.join(model_path, model, "Final", model+'_classifier.pkl'), 'rb'))

        test_tf = vectorizer.transform(test_text)
        test_tfidf = transformer.transform(test_tf)
        model_predicted_prob = classifier.predict_proba(test_tfidf)[:, 1]
        model_predicted_prob_list.append(model_predicted_prob)

# Sum over all model predictions and calculate average probabilities and
# the final labels
print("Combine results into one prediction")
test_predicted_prob = np.divide(np.sum(model_predicted_prob_list, axis=0), len(models))
test_predicted = np.where(test_predicted_prob < np.float64(threshold), 0, 1)

# Check if results folder exists for this case
if not os.path.exists(result_path):
    os.makedirs(result_path)
    
# Plot the ROC curve
fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
false_positive_rate, true_positive_rate, thresholds = roc_curve(test_labels, test_predicted_prob)
auc_value = auc(false_positive_rate, true_positive_rate)
ax.plot(false_positive_rate, true_positive_rate,
        label='ROC (AUC = {:.2f})'.format(auc_value), color='b', lw=2, alpha=.8)
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="ROC curve")
ax.legend(loc="lower right")
plt.savefig(os.path.join(result_path, "Test_ROC_{}_{}.png".format(model_name,"_".join(models))))
plt.show()

# Save the final results in a .txt file
results_file = open(os.path.join(result_path, "Test_results_{}_{}.txt".format(model_name,"_".join(models))), 
                    'w', encoding="utf8")

print("\nEnsemble consisting of {}, with a chosen threshold of: {}\n".format(models,threshold))
results_file.write("Ensemble consisting of {}, with a chosen threshold of: {}\n\n".format(models,threshold))

# Print the classification report containing precision, recall and F1-score
report = classification_report(test_labels, test_predicted, digits=3)
print(report)
results_file.write(report + "\n\n")

# Print the confusion matrix
confusion = confusion_matrix(test_labels, test_predicted)
error = confusion[0, 1] + confusion[1, 0]
print(confusion)
print("Number of errors: {}\n".format(error))
results_file.write(str(confusion))
results_file.write("\n\nNumber of errors: {}".format(error))

# Close the .txt file
results_file.close()
