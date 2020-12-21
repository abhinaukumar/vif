import numpy as np
import cv2
# from utils import structure_sim
from vif_utils import *

from scipy.io import loadmat, savemat
from scipy.stats import spearmanr, pearsonr
from scipy.optimize import curve_fit

import os
import argparse

import progressbar

parser = argparse.ArgumentParser(description="Code to generate SSIM data for Netflix Public Database")
parser.add_argument("--path", help="Path to database", required=True)
args = parser.parse_args()

path = args.path
f = loadmat('data/nflx_repo_scores.mat')
scores = f['scores'].squeeze()
scores = (scores - np.min(scores))/(np.max(scores) - np.min(scores))

ref_file_list = os.listdir(os.path.join(path, 'ref', 'rgb'))
ref_file_list = sorted([v for v in ref_file_list if v[-3:] == 'mp4'], key=lambda v: v.lower())
n_ref_files = len(ref_file_list)

dist_file_list = os.listdir(os.path.join(path, 'dis', 'rgb'))
dist_file_list = sorted([v for v in dist_file_list if v[-3:] == 'mp4'], key=lambda v: v.lower())
n_dist_files = len(dist_file_list)

widgets = [
            progressbar.ETA(),
            progressbar.Bar(),
            ' ', progressbar.DynamicMessage('file'),
            '/', progressbar.DynamicMessage('total')
            ]

sims = np.zeros((n_dist_files,))
sim_spat_alls = np.empty((n_dist_files,), dtype=object)
sim_temp_alls = np.empty((n_dist_files,), dtype=object)
i_dist = 0

kh = 5
sigma = 1.5
# win = np.exp(-0.5*(np.arange(-kh, kh+1, 1)**2/sigma**2))
# win = np.outer(win, win)
# win /= np.sum(win)

win = np.ones((2*kh+1, 2*kh+1))
win /= np.sum(win)

with progressbar.ProgressBar(max_value=n_dist_files, widgets=widgets) as bar:
    for i_ref in range(n_ref_files):
        ref_filename = ref_file_list[i_ref][:-4].split('_')[0]
        v_ref = cv2.VideoCapture(os.path.join(path, 'ref', 'rgb', ref_file_list[i_ref]))
        w = int(v_ref.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(v_ref.get(cv2.CAP_PROP_FRAME_HEIGHT))

        while(i_dist < n_dist_files and ref_filename in dist_file_list[i_dist]):
            v_dist = cv2.VideoCapture(os.path.join(path, 'dis', 'rgb', dist_file_list[i_dist]))

            y_ref_prev = np.zeros((h, w))
            y_dist_prev = np.zeros((h, w))
            spat_scores = []
            temp_scores = []
            while(v_ref.isOpened() and v_dist.isOpened()):
                ret_ref, rgb_ref = v_ref.read()
                ret_dist, rgb_dist = v_dist.read()

                if ret_ref and ret_dist:
                    y_ref = cv2.cvtColor(rgb_ref, cv2.COLOR_BGR2YUV)[:, :, 0].astype('float32')
                    y_dist = cv2.cvtColor(rgb_dist, cv2.COLOR_BGR2YUV)[:, :, 0].astype('float32')

                    # temp.append(structure_sim(y_ref, y_dist, 2, 2))
                    spat_scores.append(vif_spatial(y_ref, y_dist, win))
                    temp_scores.append(vif_spatial(y_ref - y_ref_prev, y_dist - y_dist_prev, win))
                    y_ref_prev = y_ref
                    y_dist_prev = y_dist
                else:
                    break

            sims[i_dist] = np.mean(spat_scores)*np.mean(temp_scores[1:])
            sim_spat_alls[i_dist] = np.array(spat_scores)
            sim_temp_alls[i_dist] = np.array(temp_scores[1:])

            i_dist += 1
            v_ref.set(cv2.CAP_PROP_POS_FRAMES, 0)
            bar.update(i_dist, file=i_dist + 1, total=n_dist_files)

# savemat('data/nflx_repo_stvifs.mat', {'spat_vifs': sim_spat_alls, 'temp_vifs': sim_temp_alls})
savemat('data/nflx_repo_spat_stvifs.mat', {'spat_vifs': sim_spat_alls, 'temp_vifs': sim_temp_alls})

# Fitting logistic function to SSIM
[[b0, b1, b2, b3, b4], _] = curve_fit(lambda t, b0, b1, b2, b3, b4: b0 * (0.5 - 1.0/(1 + np.exp(b1*(t - b2))) + b3 * t + b4),
                                      sims, scores, p0=0.5*np.ones((5,)), maxfev=200000)

scores_pred = b0 * (0.5 - 1.0/(1 + np.exp(b1*(sims - b2))) + b3 * sims + b4)

pcc = pearsonr(scores_pred, scores)[0]
srocc = spearmanr(scores_pred, scores)[0]
rmse = np.sqrt(np.mean((scores_pred - scores)**2))

print("PCC:", pcc)
print("SROCC:", srocc)
print("RMSE:", rmse)

# savemat('data/nflx_repo_stvifs.mat', {'spat_vifs': sim_spat_alls, 'temp_vifs': sim_temp_alls, 'pcc': pcc, 'srocc': srocc, 'rmse': rmse})
savemat('data/nflx_repo_spat_stvifs.mat', {'spat_vifs': sim_spat_alls, 'temp_vifs': sim_temp_alls, 'pcc': pcc, 'srocc': srocc, 'rmse': rmse})