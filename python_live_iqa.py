import numpy as np
import cv2
# from utils import structure_sim
from vif_utils import *

from scipy.io import loadmat
from scipy.stats import spearmanr, pearsonr
from scipy.optimize import curve_fit

import os
import argparse

import progressbar

parser = argparse.ArgumentParser(description="Code to run Structure Tensor Similarity for LIVE IQA Database")
parser.add_argument("--path", help="Path to database", required=True)
args = parser.parse_args()

path = args.path
f = loadmat('data/live_iqa_scores.mat')
scores = f['dmos'].squeeze()
scores = (scores - np.min(scores))/(np.max(scores) - np.min(scores))

f = loadmat(os.path.join(args.path, 'refnames_all.mat'))
ref_file_list = [fname[0] for fname in f['refnames_all'].squeeze()]

n_files = len(ref_file_list)

kh = 5
sigma = 1.5
win = np.exp(-0.5*(np.arange(-kh, kh+1, 1)**2/sigma**2))
win = np.outer(win, win)
win /= np.sum(win)

# win = np.ones((2*kh+1, 2*kh+1))
# win /= np.sum(win)

n_dists = {'jp2k': 227, 'jpeg': 233, 'wn': 174, 'gblur': 174, 'fastfading': 174}
widgets = [
            progressbar.ETA(),
            progressbar.Bar(),
            ' ', progressbar.DynamicMessage('file'),
            '/', progressbar.DynamicMessage('total')
            ]

sims = np.zeros((n_files,))
k = 0
with progressbar.ProgressBar(max_value=n_files, widgets=widgets) as bar:
    for dist in n_dists.keys():
        for i in range(n_dists[dist]):
            img_ref_ = cv2.imread(os.path.join(path, 'refimgs', ref_file_list[k + i]))
            img_dist_ = cv2.imread(os.path.join(path, dist, 'img'+str(i+1)+'.bmp'))
            img_ref = cv2.cvtColor(img_ref_, cv2.COLOR_BGR2YUV)[:, :, 0].astype('float32')
            img_dist = cv2.cvtColor(img_dist_, cv2.COLOR_BGR2YUV)[:, :, 0].astype('float32')

            # sims[k + i] = structure_sim(img_ref, img_dist, 2, 2)

            sims[k + i] = vif_spatial(img_ref, img_dist, win)

            bar.update(k + i, file=k + i + 1, total=n_files)

        k += n_dists[dist]

# Fitting logistic function to SSIM
[[b0, b1, b2, b3, b4], _] = curve_fit(lambda t, b0, b1, b2, b3, b4: b0 * (0.5 - 1.0/(1 + np.exp(b1*(t - b2))) + b3 * t + b4),
                                      sims, scores, p0=0.5*np.ones((5,)), maxfev=20000)

scores_pred = b0 * (0.5 - 1.0/(1 + np.exp(b1*(sims - b2))) + b3 * sims + b4)

pcc = pearsonr(scores_pred, scores)[0]
srocc = spearmanr(scores_pred, scores)[0]
rmse = np.sqrt(np.mean((scores_pred - scores)**2))

print("PCC:", pcc)
print("SROCC:", srocc)
print("RMSE:", rmse)
