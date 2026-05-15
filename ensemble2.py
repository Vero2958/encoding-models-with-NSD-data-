from sklearn.linear_model import LinearRegression
import argparse
import numpy as np
from scipy.stats import pearsonr
import os
import random

parser = argparse.ArgumentParser(description='Train and test new data')
parser.add_argument('--subject', default=2, type=int, help='Subject ID: 2 to 7')
parser.add_argument('--roi', default='FFA1', type=str, help='ROI name')
parser.add_argument('--train_size', default=None, type=int, help='Number of samples to fit')
args = parser.parse_args()
subject = args.subject
roi = args.roi
train_size = args.train_size

pred_dir = './output/nsd_ensemble/nsd_pred_responses/'
train_act = np.load(pred_dir + f'S{subject}_{roi}_train.npy')
test_act = np.load(pred_dir + f'S{subject}_{roi}_test.npy')
train_true = np.load(pred_dir + f'S{subject}_{roi}_train_true.npy')
test_true = np.load(pred_dir + f'S{subject}_{roi}_test_true.npy')

n_test_images, n_neurons = test_true.shape

le_pred = np.zeros([100, n_test_images, n_neurons])
le_acc = np.zeros(100)

for repeat in range(100):
    print('repeat %d' % repeat)
    random.seed(repeat)
    indices = random.sample(range(len(train_true)), train_size)
    X = train_act[indices]
    y = train_true[indices]
    reg = LinearRegression().fit(X, y)
    pred = reg.predict(test_act)

    corrs = [pearsonr(pred[:, v], test_true[:, v])[0] for v in range(n_neurons)]
    acc = np.nanmean(corrs)

    le_pred[repeat] = pred
    le_acc[repeat] = acc

output = {'le_pred': le_pred, 'le_acc': le_acc}
result_dir = './output/nsd_ensemble/repeat100/size%d/' % train_size
os.makedirs(result_dir, exist_ok=True)
np.save(result_dir + 'S%d_%s.npy' % (subject, roi), output)