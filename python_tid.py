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

parser = argparse.ArgumentParser(description="Code to run Structure Tensor Similarity for TID 2013 database")
parser.add_argument("--path", help="Path to database", required=True)
args = parser.parse_args()

n_ref = 25
dist_file_list = sorted(os.listdir(os.path.join(args.path, 'distorted_images')))
n_dist = len(dist_file_list) - 2

f = loadmat('data/tid13_scores.mat')
scores = f['scores'].squeeze()
scores = (scores - np.min(scores))/(np.max(scores) - np.min(scores))

widgets = [
            progressbar.ETA(),
            progressbar.Bar(),
            ' ', progressbar.DynamicMessage('file'),
            '/', progressbar.DynamicMessage('total')
          ]

sims = np.zeros((n_dist,))

kh = 5
sigma = 1.5
# win = np.exp(-0.5*(np.arange(-kh, kh+1, 1)**2/sigma**2))
# win = np.outer(win, win)
# win /= np.sum(win)

win = np.ones((2*kh+1, 2*kh+1))
win /= np.sum(win)

k = 0
with progressbar.ProgressBar(max_value=n_dist, widgets=widgets) as bar:
    for i in range(1, n_ref + 1):
        img_ref_ = cv2.imread(os.path.join(args.path, 'reference_images', 'i'+str(i).zfill(2)+'.bmp'))
        if img_ref_ is None:
            img_ref_ = cv2.imread(os.path.join(args.path, 'reference_images', ('i'+str(i).zfill(2)+'.bmp').upper()))

        dist_files = [f for f in dist_file_list if 'i'+str(i).zfill(2) in f or ('i'+str(i).zfill(2)).upper() in f]
        dist_files.sort(key=lambda string: string.lower())

        for dist_file in dist_files:

            img_dist_ = cv2.imread(os.path.join(args.path, 'distorted_images', dist_file))
            if img_dist_ is None:
                img_dist_ = cv2.imread(os.path.join(args.path, 'distorted_images', dist_file.upper()))

            img_ref = cv2.cvtColor(img_ref_, cv2.COLOR_BGR2YUV)[:, :, 0].astype('float32')
            img_dist = cv2.cvtColor(img_dist_, cv2.COLOR_BGR2YUV)[:, :, 0].astype('float32')

            # sims[k] = structure_sim(img_ref, img_dist, 2, 2)
            sims[k] = vif_spatial(img_ref, img_dist, win)
            k += 1
            bar.update(k, file=k, total=n_dist)

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
