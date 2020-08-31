### Supplementary materials for "Automatic classification of literature in systematic reviews on food safety using machine learning"

#### L.M. van den Bulk, Y. Bouzembrak, A. Gavai, N. Liu, L.J. van den Heuvel, H.J.P. Marvin

The code was implemented using Python 3.7. All package requirements can be found in "requirements.txt" above and can be installed using `pip install --ignore-installed -r requirements.txt` after cloning the repository. We recommend using a conda environment if you are installing from scratch. The code will run out of the box. Note that if you want the tensorflow package (used for the neural network models) to run on your GPU, you will need to install additional [software](https://www.tensorflow.org/install/gpu).

The "Models" and "Results" folders are purposefully empty, these will be filled automatically when running the code. The training scripts in which models are trained on the train data should be run before the corresponding test scripts can be run successfully. The crossvalidation scripts were used to find the optimal parameters for the models, these parameters are used in the training scripts. Note that the final LSTM model is generated by the crossvalidation script and that there does not exist a separate training script, as it uses a validation set for early stopping to prevent overfitting and therefore can't train on the entire train set.
