import numpy as np
import cv2
# from utils import structure_sim
from vif_utils import *

from scipy.io import loadmat, savemat
from scipy.stats import spearmanr, pearsonr
from scipy.optimize import curve_fit

import pandas as pd
import os
import argparse

import progressbar

parser = argparse.ArgumentParser(description="Code to generate SSIM data for LIVE IQA Database")
parser.add_argument("--path", help="Path to database", required=True)
args = parser.parse_args()

f = loadmat('data/live_vqa_scores.mat')
scores = f['scores'].squeeze()
scores = (scores - np.min(scores))/(np.max(scores) - np.min(scores))

df = pd.read_csv(os.path.join(args.path, 'live_video_quality_seqs.txt'), header=None, engine='python')
file_list = df.values[:, 0]

refs = ["pa", "rb", "rh", "tr", "st", "sf", "bs", "sh", "mc", "pr"]
fps = [25, 25, 25, 25, 25, 25, 25, 50, 50, 50]
fps = [str(f) + 'fps' for f in fps]

n_refs = len(refs)

widgets = [
            progressbar.ETA(),
            progressbar.Bar(),
            ' ', progressbar.DynamicMessage('file'),
            '/', progressbar.DynamicMessage('total')
            ]

sims = np.zeros((n_refs*15,))
sim_spat_alls = np.empty((n_refs*15,), dtype=object)
sim_temp_alls = np.empty((n_refs*15,), dtype=object)
k = 0

kh = 5
sigma = 1.5
# win = np.exp(-0.5*(np.arange(-kh, kh+1, 1)**2/sigma**2))
# win = np.outer(win, win)
# win /= np.sum(win)

win = np.ones((2*kh+1, 2*kh+1))
win /= np.sum(win)

with progressbar.ProgressBar(max_value=n_refs*15, widgets=widgets) as bar:
    for i_ref, ref in enumerate(refs):
        v_ref = cv2.VideoCapture(os.path.join(args.path, 'videos', ref + '_Folder', 'rgb', ref + '1' + '_' + fps[i_ref] + '.mp4'))
        w = int(v_ref.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(v_ref.get(cv2.CAP_PROP_FRAME_HEIGHT))
        for i_dist in range(2, 17):
            v_dist = cv2.VideoCapture(os.path.join(args.path, 'videos', ref + '_Folder', 'rgb', ref + str(i_dist) + '_' + fps[i_ref] + '.mp4'))
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

            sims[k] = np.mean(spat_scores)*np.mean(temp_scores[1:])
            sim_spat_alls[k] = np.array(spat_scores)
            sim_temp_alls[k] = np.array(temp_scores[1:])
            k += 1
            v_ref.set(cv2.CAP_PROP_POS_FRAMES, 0)
            bar.update(i_ref*15 + i_dist - 2, file=i_ref*15+i_dist-1, total=n_refs*15)

# savemat('data/live_vqa_stvifs.mat', {'spat_vifs': sim_spat_alls, 'temp_vifs': sim_temp_alls})
# savemat('data/live_vqa_ifcs.mat', {'spat_ifcs': sim_spat_alls})
savemat('data/live_vqa_spat_stvifs.mat', {'spat_vifs': sim_spat_alls, 'temp_vifs': sim_temp_alls})

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

# savemat('data/live_vqa_stvifs.mat', {'spat_vifs': sim_spat_alls, 'temp_vifs': sim_temp_alls, 'pcc': pcc, 'srocc': srocc, 'rmse': rmse})
# savemat('data/live_vqa_stvifs.mat', {'spat_vifs': sim_spat_alls, 'pcc': pcc, 'srocc': srocc, 'rmse': rmse})
savemat('data/live_vqa_spat_stvifs.mat', {'spat_vifs': sim_spat_alls, 'pcc': pcc, 'srocc': srocc, 'rmse': rmse})