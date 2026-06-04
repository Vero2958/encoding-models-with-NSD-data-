# Ensemble model
This readme file presents main facts about the project and how to use the and run the files in this repository.

## Introduction
The code in this repository in part of a project on encoding models of some areas of the visual cortex, specifically V1v, PPA, EBA and FFA1. The aim of this project is building an encoding model with the minimum possible amount of data in such a way that the accuracy of the model is comparable to that of another "baseline" model utilizing all the data available for a given subject. Part of the code in this repo is based on the work of Gu et al (2022) "Personalized visual encoding model construction with small data". The results reported in this paper might have a limitation regarding spatial resolution: the accuracy of their model is only comparable to the accuracy of the complete model when predicting the average activation of a ROI, not its voxel-wise activation. 

### Files
1. `dataloaders.py`: Defines the NSDdataset PyTorch Dataset. Loads images and ROI neural responses from disk for a given subject and split (train/val/test). If train_size is specified, randomly subsamples the training data with a subject-seeded RNG for reproducibility. Images are transposed and normalized with ImageNet stats. Returns (image, response) pairs.
2. `individual_model.py`: This script trains individual ResNet50-based encoding models, for a given subject and ROI, it sets up a FeatCore (ResNet50 backbone) + SimpleLinear readout as an Encoder, then trains it end-to-end with Adam and a masked MSE loss to predict neural responses from images. Training uses gradient accumulation and an early stopping mechanism: every 100 iterations it evaluates Pearson correlation on a validation set, saves the checkpoint if it improves, and stops if there's no improvement for patience=20 checks
3. `ensemble.py`: part one of the ensemble model, it loads pre-trained ResNet50-based encoder models, one per NSD subject (excluding the target subject), then runs both training and test images through each model, collects the predicted neural activations, and stacks them horizontally into a combined feature matrix. The true neural responses are also collected. Saves everything as a .npy file.
4. `ensemble2.py`: part 2 of the ensemble, it loads the files that were previously saved in part 1 and then runs 100 repetitions: each time it randomly samples the train_size training examples, fits a linear regression on the ensemble features, predicts the responses on the test set, and computes per-voxel Pearson correlation between predicted and true responses (averaged across neurons as accuracy). The predictions and accuracy scores across all 100 repeats are saved as a .npy file.
5. `read output.py`: this code is used to access the content of the files saved by part 2 of the ensemble, it prints accuracy and standard deviation.
6. `individual_finetune.py`: Same as individual_model.py but adds a --retrain flag. When enabled, the readout layer is initialized with the mean weights and biases of the other subjects' readout parameters (loaded from a precomputed .npy file), before fine-tuning on the target subject. Checkpoint path is also adjusted accordingly.
7. `models.py`: Defines the model architecture. Contains the following:
* FeatCore: wraps a pretrained ResNet50 (or VGG19-BN) backbone with the classification head removed, optionally freezing weights.
* SimpleLinear: is a linear readout that applies max pooling over spatial dimensions followed by a fully connected layer.
* Encoder: combines core and readout into a single forward pass.
8. `usefunction.py`: Contains utility functions used during training and evaluation:
* masked_MSEloss: computes MSE loss ignoring missing/invalid neural responses (flagged as values below -900).
* full_objective: wraps the model forward pass and loss computation.
* compute_predictions: runs inference over a dataloader and returns stacked true and predicted responses.
* compute_scores: computes mean Pearson correlation across neurons, skipping invalid entries (-999).
* save_checkpoint: saves model and optimizer state to disk, keeping both the latest and best checkpoint.

### Pipeline:
1. Training (`individual_model.py`): A ResNet50 encoder is fine-tuned per subject and ROI to predict neural responses from images.
2. Ensemble feature extraction (`ensemble.py`): For a target subject, all other subjects' trained models are loaded and used to extract predictions on train and test images, which are stacked into a combined feature matrix.
3. Linear regression and Evaluation (`ensemble2.py`): A linear regression is fit on top of the ensemble features to predict the target subject's neural responses. This is repeated 100 times with random subsamples of varying train_size to assess sample efficiency.
4. Support modules: `models.py` defines the architecture, `dataloaders.py` handles data loading, `usefunction.py` provides loss, scoring, and checkpoint utilities.

## Data
The data for this project comes from the NSD dataset (https://naturalscenesdataset.org/). Specifically, we are using the preprocessed version of the data, in its version number 3, which is the least noisy one and therefore the most appropriate in the contex of the scope of this project. To run the code we have prepared and made available on kaggle a version of the data in the correct format and already split in test, train and validation sets, which can be found at the following links:
* S1: https://www.kaggle.com/datasets/veronicascozzi/nsd-s1-test-val-train
* S2: https://www.kaggle.com/datasets/veronicascozzi/nsd-s2-test-val-train
* S3: https://www.kaggle.com/datasets/veronicascozzi/nsd-s3-test-val-train
* S4: https://www.kaggle.com/datasets/veronicascozzi/nsd-s4-test-val-train
* S5: https://www.kaggle.com/datasets/veronicascozzi/nsd-s5-test-val-train
* S6: https://www.kaggle.com/datasets/veronicascozzi/nsd-s6-test-val-train
* S7: https://www.kaggle.com/datasets/veronicascozzi/nsd-s7-test-val-train
* S8: https://www.kaggle.com/datasets/veronicascozzi/nsd-s8-test-val-train

These datasets contain 6 files each, which are arrays of images and corresponding responses for each one of the sets. For the images presented more than once, the response was averaged across trials in order to both further increase the SNR and decrease the training time of the individual model. The total amount of data therefor consists of about 10000 image-response pairs, of which 500 make up the validation set, the 766 image that were shared across subject and seen at least twice by all of the subject, together with their corresponding responses, make up the test set, while the remaining at most 8500 make up the train set.

For those who wish to run this code on a cloud (highly recomended), for easier and faster access the same files contained in this directory have already been uploaded on kaggle at the following link:
* https://www.kaggle.com/datasets/veronicascozzi/nsd-research-files

## Usage:
1. **Run individual models training**
   First of all, we need to run the training for the 20K (10K in our case) model, which can be done by using the following command:
   ```bash
   python individual_model.py --subject *subject_number1-8* --roi *roi_name*
   ```
   This must be done for all ROIs and all subjects.

2. **Verify checkpoints directory**
   After this has been done, you'll have a directory called `ckpt_ROI`, containing the checkpoint for each ROI for every subject. Make sure that the name of these files is `best_resnet50_roiname_finetune_linear.pth.tar` or `last_resnet50_roiname_finetune_linear.pth.tar`. You won't need the `last` ones, but the code saves both.

3. **Run ensemble.py**
   Now the checkpoints can be used to train and test the ensemble model. Run `ensemble.py` using the following command:
   ```bash
   python ensemble.py --subject *subject_number1-8* --roi *roi_name*
   ```
   After this finishes running, the necessary files to run `ensemble2.py` should be found in the directory `./output/nsd_ensemble/nsd_pred_responses`.

4. **Run ensemble2.py**
   At this point, `ensemble2.py` can be run using the following command:
   ```bash
   python ensemble2.py --subject *subject_number1-8* --roi *roi_name* --train_size *train_size*
   ```
   The output is saved in the directory `./output/nsd_ensemble/repeat100/size%d/`.

5. **Read the output**
   Lastly, the output can be read by using the script called `read_output.py`. Simply change the filename in the `np.load()` command. For example, change it from `./output/nsd_ensemble/repeat100/size300/S2_V1v.npy` to `./output/nsd_ensemble/repeat100/size300/S3_FFA1.npy`.

## Project structure
Once you have downloaded all data and scripts, the project directory should have the following structure:
```text
+---.idea
|   \---inspectionProfiles
+---data
|   +---S1
|   +---S2
|   +---S3
|   +---S4
|   +---S5
|   +---S6
|   +---S7
|   \---S8
+---venv
+---dataloaders.py
+---ensemble.py
+---ensemble2.py
+---individual_finetune.py
+---individual_model.py
+---models.py
+---read_output.py
+---usefuncs.py
+---README.md
```

After completing step 1 of the usage, you should have an additional directory ckpt_ROI containing the checkpoints generated by the training, with the following structure:

```text
+---ckpt_ROI
|   +---S1
|   +---S2
|   +---S3
|   +---S4
|   +---S5
|   +---S6
|   +---S7
|   \---S8
```
with each subdirectory containing best and last checkpoints of the corresponding subject for each roi (8 files per subdirectory).

After completing step 3, you should have a further directory with the following structure:

```text
+---output
|   \---nsd_ensemble
|       +---nsd_pred_responses
```

with `nsd_pred_responses` containing the files generated by ensemble.py for each subject and for each roi,` with name S{subject}_{roi}_train.npy`

Lastly, after completing step 4, this directory should further expand and should look like the following:

```text
+---output
|   \---nsd_ensemble
|       +---nsd_pred_responses
|       \---repeat100
|           \---size300
```

with `size300` containing the output of ensemble.py with name `S{subject}_{roi}.npy`

## Notes: 
1. that subject numebers are 1-indexed and therefore go from 1 to 8, not from 0 to 7.
2. roi names are V1v, FFA1, EBA and PPA, respect caps.
3. as train_size for ensemble2.py please use 300.
4. The code utilizes **NVIDIA CUDA**. Therefore, the code **cannot be executed locally** unless your machine is equipped with an NVIDIA GPU. In case your device doesn't feature one, you can run this code in the cloud using one of the following platforms:
* Google Colab: Create a new notebook and enable the GPU by going to *Runtime* > *Change runtime type* > Select **T4 GPU** (or higher).
* Kaggle: Create a notebook, open the right-side panel under *Settings* > *Accelerator*, and select **GPU T4 x2** or **GPU P100**.
