# Ensemble model
This readme file presents main facts about the project and how to use the and run the files in this repository.

## Introduction
The code in this repository in part of a project on encoding models of the visual cortex. The aim of this project is building an encoding model with the minimum possible amount of data in such a way that the accuracy of the model is comparable to that of another "baseline" model utilizing all the data available for a given subject. Part of the code in this repo is based on the work of Gu et al (2022) "Personalized visual encoding model construction with small data". The results reported in this paper might have a limitation regarding spatial resolution: the accuracy of their model is only comparable to the accuracy of the complete model when predicting the average activation of a ROI, not its voxel-wise activation. 

### Files
1. dataloaders.py: Defines the NSDdataset PyTorch Dataset. Loads images and ROI neural responses from disk for a given subject and split (train/val/test). If train_size is specified, randomly subsamples the training data with a subject-seeded RNG for reproducibility. Images are transposed and normalized with ImageNet stats. Returns (image, response) pairs.
2. ensemble.py: part one of the ensemble model, it loads pre-trained ResNet50-based encoder models, one per NSD subject (excluding the target subject), then runs both training and test images through each model, collects the predicted neural activations, and stacks them horizontally into a combined feature matrix. The true neural responses are also collected. Saves everything as a .npy file.
3. ensemble2.py: part 2 of the ensemble, it loads the files that were previously saved in part 1 and then runs 100 repetitions: each time it randomly samples the train_size training examples, fits a linear regression on the ensemble features, predicts the responses on the test set, and computes per-voxel Pearson correlation between predicted and true responses (averaged across neurons as accuracy). The predictions and accuracy scores across all 100 repeats are saved as a .npy file.
4. read output.py: this code is used to access the content of the files saved by part 2 of the ensemble, it prints accuracy and standard deviation.
5. individual_model.py: This script trains individual ResNet50-based encoding models, for a given subject and ROI, it sets up a FeatCore (ResNet50 backbone) + SimpleLinear readout as an Encoder, then trains it end-to-end with Adam and a masked MSE loss to predict neural responses from images. Training uses gradient accumulation and an early stopping mechanism: every 100 iterations it evaluates Pearson correlation on a validation set, saves the checkpoint if it improves, and stops if there's no improvement for patience=20 checks
6. individual_finetune.py: Same as individual_model.py but adds a --retrain flag. When enabled, the readout layer is initialized with the mean weights and biases of the other subjects' readout parameters (loaded from a precomputed .npy file), before fine-tuning on the target subject. Checkpoint path is also adjusted accordingly.
7. models.py: Defines the model architecture. Contains the following:
1.FeatCore: wraps a pretrained ResNet50 (or VGG19-BN) backbone with the classification head removed, optionally freezing weights.
2.SimpleLinear: is a linear readout that applies max pooling over spatial dimensions followed by a fully connected layer.
3.Encoder: combines core and readout into a single forward pass.
8. usefunction.py: Contains utility functions used during training and evaluation:
1.masked_MSEloss: computes MSE loss ignoring missing/invalid neural responses (flagged as values below -900).
2.full_objective: wraps the model forward pass and loss computation.
3.compute_predictions: runs inference over a dataloader and returns stacked true and predicted responses.
4.compute_scores: computes mean Pearson correlation across neurons, skipping invalid entries (-999).
5.save_checkpoint: saves model and optimizer state to disk, keeping both the latest and best checkpoint.

### Pipeline:
1. Training (train.py, train_retrain.py): A ResNet50 encoder is fine-tuned per subject and ROI to predict neural responses from images. train_retrain.py optionally initializes the readout from the mean weights of other subjects before fine-tuning.
2. Ensemble feature extraction (ensemble.py): For a target subject, all other subjects' trained models are loaded and used to extract predictions on train and test images, which are stacked into a combined feature matrix.
3. Linear regression and Evaluation (ensemble2.py): A linear regression is fit on top of the ensemble features to predict the target subject's neural responses. This is repeated 100 times with random subsamples of varying train_size to assess sample efficiency.
4. Support modules: models.py defines the architecture, dataloaders.py handles data loading, usefuncs.py provides loss, scoring, and checkpoint utilities.

## Data
The data for this project comes from the NSD dataset (https://naturalscenesdataset.org/). Specifically, we are using the preprocessed version of the data, in its version number 3, which is the least noisy one and therefore the most appropriate in the contex of the scope of this project. To run the code we have prepared and made available on kaggle a version of the data in the correct format and already split in test, train and validation sets, which can be found at the following links:
1. S1: https://www.kaggle.com/datasets/veronicascozzi/nsd-s1-test-val-train
2. S2: https://www.kaggle.com/datasets/veronicascozzi/nsd-s2-test-val-train
3. S3: https://www.kaggle.com/datasets/veronicascozzi/nsd-s3-test-val-train
4. S4: https://www.kaggle.com/datasets/veronicascozzi/nsd-s4-test-val-train
5. S5: https://www.kaggle.com/datasets/veronicascozzi/nsd-s5-test-val-train
6. S6: https://www.kaggle.com/datasets/veronicascozzi/nsd-s6-test-val-train
7. S7: https://www.kaggle.com/datasets/veronicascozzi/nsd-s7-test-val-train
8. S8: https://www.kaggle.com/datasets/veronicascozzi/nsd-s8-test-val-train

These datasets contain 6 files each, which are arrays of images and corresponding responses for each one of the sets. For the images presented more than once, the response was averaged across trials in order to both further increase the SNR and decrease the training time of the individual model. The total amount of data therefor consists of about 10000 image-response pairs, of which 500 make up the validation set, the 766 image that were shared across subject and seen at least twice by all of the subject, together with their corresponding responses, make up the test set, while the remaining at most 8500 make up the train set.

## Project structure
add a description of the structure the local directory should have once one has downloaded all data and all scripts.

## Usage
add description and commands to run to succesfully use the code.
