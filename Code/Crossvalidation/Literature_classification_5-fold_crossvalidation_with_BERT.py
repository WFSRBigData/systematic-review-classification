# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 11:27:26 2020

@author: bulk007

Code to train a distilled BERT model using 5-fold crossvalidation to classify
papers in a specific topic as "Relevant" or "Not relevant" based on their title
and abstract. The used literature data has been selected from academic databases
based on keywords as labeled by human experts.
"""

# Import needed packages
import os
import re
import numpy as np
from numpy import interp
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsRegressor
import silence_tensorflow.auto
import tensorflow as tf
import ktrain
from ktrain import text as ktrain_text
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Set random seed to be able to compare across models
np.random.seed(1)

# Set general parameters
case_study = 'Cereals' # Choose 'Cereals' or 'Leafy Greens'
model_name = 'BERT'
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

    # A distilled model of BERT is used with less parameters as we do not have
    # a lot of data. Preprocessing from text to numeric data is done in the
    # code below in a way designed for the BERT algorithm.
    print("Preprocessing data")
    tf.autograph.set_verbosity(0)
    bert_model = 'distilbert-base-uncased'
    t = ktrain_text.Transformer(bert_model, maxlen=500, class_names=[0, 1])
    train_preprocessed = t.preprocess_train(train_text, train_labels)
    val_preprocessed = t.preprocess_test(val_text, val_labels)

    # In order to create a more balanced dataset SMOTE can be applied as
    # oversampling technique. Depending on the imbalance between 'Relevant' and
    # 'Not relevant' using SMOTE might be necessary
    if(smote):
        print("Performing SMOTE")
        train_preprocessed_text = train_preprocessed.x.reshape(train_preprocessed.x.shape[0],
                                                        train_preprocessed.x.shape[1]*train_preprocessed.x.shape[2])
        train_preprocessed_labels = (train_preprocessed.y[:,1] == 1).astype(int)
        sm = SMOTE(random_state=42, k_neighbors=3)
        train_preprocessed_text, train_preprocessed_labels = sm.fit_sample(train_preprocessed_text, train_preprocessed_labels)
        train_preprocessed_text = train_preprocessed_text.reshape(train_preprocessed_text.shape[0],
                                                        train_preprocessed.x.shape[1], train_preprocessed.x.shape[2])
        train_preprocessed_labels = np.eye(2)[train_preprocessed_labels]
        train_preprocessed.x = train_preprocessed_text
        train_preprocessed.y = train_preprocessed_labels

    # Use a nearest neighbors algorithm to augment the data to expand the train set by 20%
    if(data_augmentation):
        print("Performing data augmentation")
        augmented_train_preprocessed = train_preprocessed.x.copy()
        augmented_train_labels = train_preprocessed.y.copy()   
        train_preprocessed_text = train_preprocessed.x.reshape(train_preprocessed.x.shape[0],
                                                        train_preprocessed.x.shape[1]*train_preprocessed.x.shape[2])
        train_preprocessed_labels = train_preprocessed.y
        knn = KNeighborsRegressor(4, 'distance').fit(train_preprocessed_text, train_preprocessed_labels)
        shuffled_indexes = list(range(len(train_preprocessed_text)))
        np.random.shuffle(shuffled_indexes)

        # Augment 20% of the train data and add it to the original set
        for index in shuffled_indexes[0:int(len(train_preprocessed_text)/5)]:
            datapoint_text = np.reshape(train_preprocessed_text[index], (1, -1))
            datapoint_label = train_preprocessed_labels[index].reshape(1, train_preprocessed_labels[index].shape[0])
            neighbor = knn.kneighbors(datapoint_text, return_distance=False)
            random_neighbor = np.random.randint(1, 4)
            difference = train_preprocessed_text[neighbor[0][random_neighbor]] - datapoint_text
            gap = np.random.rand(1)[0]
            new_point = (datapoint_text + difference*gap).astype(int)
            new_point = new_point.reshape(1,
                                          train_preprocessed.x.shape[1],
                                          train_preprocessed.x.shape[2])
            augmented_train_preprocessed = np.append(augmented_train_preprocessed, new_point, axis=0)
            augmented_train_labels = np.append(augmented_train_labels, datapoint_label, axis=0)

        train_preprocessed.x = augmented_train_preprocessed
        train_preprocessed.y = augmented_train_labels


    # Initialize BERT classifier and fit it on the preprocessed training and
    # data validation data
    print("Training the model")
    model = t.get_classifier()
    learner = ktrain.get_learner(model, train_data=train_preprocessed,
                                 val_data=val_preprocessed, batch_size=1)
    learner.fit_onecycle(5e-5, 3)

    # Save the model for this fold
    print("Saving model and testing on validation set")
    predictor = ktrain.get_predictor(learner.model, preproc=t)
    if not os.path.exists(os.path.join(save_path, model_name, "Fold_{}".format(i))):
        os.makedirs(os.path.join(save_path, model_name, "Fold_{}".format(i)))
    predictor.save(os.path.join(save_path, model_name, "Fold_{}".format(i), 'BERT_model'))

    # Test the model on the test data
    pred_val_labels = predictor.predict_proba(val_text)[:, 1]

    val_labels_list = val_labels_list + list(val_labels)
    val_predicted_prob_list = val_predicted_prob_list + list(pred_val_labels)

    # Plot the ROC curve and save the true positive rate and the area under the curve
    false_positive_rate, true_positive_rate, thresholds = roc_curve(val_labels, pred_val_labels)
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