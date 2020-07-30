# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 16:56:58 2020

@author: bulk007

Code to test the final SVM model on the future set to classify papers in a
specific topic as "Relevant" or "Not relevant" based on their title and abstract.
The used literature data has been selected from academic databases based on
keywords as labeled by human experts.
"""

# Import needed packages
import os
import re
import numpy as np
from sklearn.metrics import plot_roc_curve, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pickle

# Set random seed to be able to compare across models
np.random.seed(1)

# Set general parameters
case_study = 'Cereals' # Choose 'Cereals' or 'Leafy Greens'
model_name = 'SVM'
data_path = os.path.join(os.getcwd(), '../../..', 'Data', case_study)
model_path = os.path.join(os.getcwd(), '../../..', 'Models', case_study)
result_path = os.path.join(os.getcwd(), '../../..', 'Results', case_study, 'Future_set')
threshold = 0.5 # Set the threshold you want to use

# The used literature was supplied as Endnote library and was exported as a
# .txt file in BibTeX style per label (i.e. Relevant and Not Relevant). We
# load the .txt file, parse them and transform into a dataset consisting of the
# title+abstract as input and the label as output
with open(os.path.join(data_path, "Future_set", case_study.replace(" ", "") + '_1920_Relevant.txt'),
          'r', encoding="utf8") as file:
    papers_relevant = file.read()
    papers_relevant = re.split(r'\@.*?{', papers_relevant)[1:]

with open(os.path.join(data_path, "Future_set", case_study.replace(" ", "") + '_1920_NotRelevant.txt'),
          'r', encoding="utf8") as file:
    papers_not_relevant = file.read()
    papers_not_relevant = re.split(r'\@.*?{', papers_not_relevant)[1:]

test_text = []
test_labels = []

for entry in papers_relevant:
    if("title = {" in entry):
        test_labels.append(1)
        paper_text = entry.split("title = {")[1].split("},\n")[0] + ". "
        if("abstract = {" in entry):
            abstract = entry.split("abstract = {")[1].split("},\n")[0]
            paper_text = paper_text + abstract
        test_text.append(paper_text)

for entry in papers_not_relevant:
    if("title = {" in entry):
        test_labels.append(0)
        paper_text = entry.split("title = {")[1].split("},\n")[0] + ". "
        if("abstract = {" in entry):
            abstract = entry.split("abstract = {")[1].split("},\n")[0]
            paper_text = paper_text + abstract
        test_text.append(paper_text)

# Perform prepocessing on the text to remove special characters and numbers.
# After this remove the double spaces that might be added because of that step
test_text = [re.sub(r"[!\"’\'#$%&©±()×*+,-./:;<=>≥≤»→?@^_`{|}~\[\]\\0-9]", "", i) for i in test_text]
test_text = [re.sub(r"\s+", " ", i) for i in test_text]
# Load the vectorizer, transformer and classifer from memory
vectorizer = pickle.load(open(os.path.join(model_path, model_name, "Final", model_name+'_vectorizer.pkl'), 'rb'))
transformer = pickle.load(open(os.path.join(model_path, model_name, "Final", model_name+'_transformer.pkl'), 'rb'))
classifier = pickle.load(open(os.path.join(model_path, model_name, "Final", model_name+'_classifier.pkl'), 'rb'))

# Make predictions on the test set
test_tf = vectorizer.transform(test_text)
test_tfidf = transformer.transform(test_tf)
test_predicted_prob = classifier.predict_proba(test_tfidf)[:, 1]

# Determine the final predicted class labels for this model
test_predicted = np.where(test_predicted_prob < np.float64(threshold), 0, 1)

# Check if results folder exists for this case
if not os.path.exists(result_path):
    os.makedirs(result_path)

# Plot the ROC curve
fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
roc = plot_roc_curve(classifier, test_tfidf, test_labels, color='b',
                     name='ROC', lw=2, alpha=.8, ax=ax)
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="ROC curve")
ax.legend(loc="lower right")
plt.savefig(os.path.join(result_path, "Future_set_ROC_{}.png".format(model_name)))
plt.show()

# Save the final results in a .txt file
results_file = open(os.path.join(result_path, "Future_set_results_{}.txt".format(model_name)), 
                    'w', encoding="utf8")

print("\n{}, with a chosen threshold of: {}\n".format(model_name,threshold))
results_file.write("{}, with a chosen threshold of: {}\n\n".format(model_name,threshold))

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
