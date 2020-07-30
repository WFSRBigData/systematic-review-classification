# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 10:21:35 2020

@author: bulk007

Code to train a Gradient Boosting model using 5-fold crossvalidation to classify
papers in a specific topic as "Relevant" or "Not relevant" based on their title
and abstract. The used literature data has been selected from academic databases
based on keywords as labeled by human experts.
"""

# Import needed packages
import os
import re
import numpy as np
from scipy import sparse
from numpy import interp
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import plot_roc_curve, auc, classification_report, confusion_matrix
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import pickle

# Set random seed to be able to compare across models
np.random.seed(1)

# Set general parameters
case_study = 'Cereals' # Choose 'Cereals' or 'Leafy Greens'
model_name = 'GradientBoosting'
smote = False  # Set this to True in case of a big class imbalance
data_augmentation = False  # Set this to True in case there is little data available
data_path = os.path.join(os.getcwd(), '../..', 'Data', case_study)
save_path = os.path.join(os.getcwd(), '../..', 'Models', case_study)
result_path = os.path.join(os.getcwd(), '../..', 'Results', case_study, 'Crossvalidation')

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

    print("Fold {} of outer crossvalidation".format(i))

    # Split the train and validation set
    train_text = np.take(train_val_text, train_indices)
    train_labels = np.take(train_val_labels, train_indices)

    val_text = np.take(train_val_text, val_indices)
    val_labels = np.take(train_val_labels, val_indices)

    # We have to transform our data into numeric vectors for Gradient Boosting
    # to work, we will first create a document-term matrix for the training set
    print("Vectorizing data and applying TFIDF")
    count_vectorizer = CountVectorizer(lowercase=True, stop_words='english',
                                       ngram_range=(1, 1), tokenizer=None)
    trained_vectorizer = count_vectorizer.fit(train_text)
    train_tf = trained_vectorizer.transform(train_text)
    val_tf = trained_vectorizer.transform(val_text)

    # We will divide the counts by frequency to not favour longer texts
    tfidf_transformer = TfidfTransformer()
    trained_transformer = tfidf_transformer.fit(train_tf)
    train_tfidf = trained_transformer.transform(train_tf)
    val_tfidf = trained_transformer.transform(val_tf)

    # In order to create a more balanced dataset SMOTE can be applied as
    # oversampling technique. Depending on the imbalance between 'Relevant' and
    # 'Not relevant' using SMOTE might be necessary
    if(smote):
        print("Performing SMOTE")
        sm = SMOTE(random_state=42, k_neighbors=3)
        train_tfidf, train_labels = sm.fit_sample(train_tfidf, train_labels)

    # Use a nearest neighbors algorithm to augment the data to expand the train set
    if(data_augmentation):
        print("Performing data augmentation")
        train_tfidf = train_tfidf.toarray()
        augmented_train_tfidf = list(train_tfidf.copy())
        augmented_train_labels = list(train_labels.copy())
        knn = KNeighborsRegressor(4, 'distance').fit(train_tfidf, train_labels)
        shuffled_indexes = list(range(len(augmented_train_tfidf)))
        np.random.shuffle(shuffled_indexes)

        # Augment 20% of the train data and add it to the original set
        for index in shuffled_indexes[0:int(len(augmented_train_tfidf)/5)]:
            datapoint_text = np.reshape(augmented_train_tfidf[index], (1, -1))
            datapoint_label = augmented_train_labels[index]
            neighbor = knn.kneighbors(datapoint_text, return_distance=False)
            random_neighbor = np.random.randint(1, 4)
            difference = train_tfidf[neighbor[0][random_neighbor]] - datapoint_text
            gap = np.random.rand(1)[0]
            new_point = datapoint_text + difference*gap
            augmented_train_tfidf = np.append(augmented_train_tfidf, new_point, axis=0)
            augmented_train_labels.append(datapoint_label)

        train_tfidf = sparse.csr_matrix(augmented_train_tfidf)
        train_labels = augmented_train_labels

    # Initialize Gradient Boosting classifier and fit it on the tf-idf training data
    print("Training the model")
    classifier = GradientBoostingClassifier(n_estimators=2000, learning_rate=0.01,
                                            max_depth=1, max_features=None,
                                            random_state=42)
    trained_classifier = classifier.fit(train_tfidf, train_labels)

    # Save the vectorizer, transformer and model for this fold
    print("Saving model and testing on validation set")
    if not os.path.exists(os.path.join(save_path, model_name, "Fold_{}".format(i))):
        os.makedirs(os.path.join(save_path, model_name, "Fold_{}".format(i)))
    pickle.dump(trained_vectorizer, open(os.path.join(save_path, model_name, "Fold_{}".format(i), model_name+'_vectorizer.pkl'), 'wb'))
    pickle.dump(trained_transformer, open(os.path.join(save_path, model_name, "Fold_{}".format(i), model_name+'_transformer.pkl'), 'wb'))
    pickle.dump(trained_classifier, open(os.path.join(save_path, model_name, "Fold_{}".format(i), model_name+'_classifier.pkl'), 'wb'))

    # Make predictions on the validation set
    val_prob_predicted = trained_classifier.predict_proba(val_tfidf)[:, 1]

    # Save the ROC curves of this fold and remember the value of the AUC and
    # the elements of the confusion matrix
    roc = plot_roc_curve(classifier, val_tfidf, val_labels,
                         name='ROC fold {}'.format(i+1),
                         alpha=0.3, lw=1, ax=ax)

    # To calculate the mean over the folds, we need a consistent number of
    # thresholds per fold, which we force by interpolation over a standard 
    # length of 100 values    
    mean_false_positive_rates = np.linspace(0, 1, 100)
    true_positive = interp(mean_false_positive_rates, roc.fpr, roc.tpr)
    true_positive[0] = 0  # Set the initial value to 0 as it should start there
    true_positive[-1] = 1  # Set the last value to 1 as it should end there
    true_positive_rates.append(true_positive)
    area_under_curves.append(roc.roc_auc)
    val_labels_list = val_labels_list + list(val_labels)
    val_predicted_prob_list = val_predicted_prob_list + list(val_prob_predicted)

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
plt.savefig(os.path.join(result_path, "5-fold_crossvalidation_ROC_{}.png".format(model_name)))
plt.show()

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
