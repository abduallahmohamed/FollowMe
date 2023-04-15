import torch
from torch.utils.data import Dataset
import os
import numpy as np
import glob
import pickle
from tqdm import tqdm


class VFDataSet(Dataset):
    def __init__(self,
                 dir,
                 obs_len=10,
                 pred_len=30,
                 srrounding_limit=1000,
                 moving_window_size=10):

        args_str = dir + str(obs_len) + str(pred_len) + str(
            srrounding_limit) + str(moving_window_size)
        pkl_path = './pkls/' + args_str.replace("/", "_") + 'intersection.pkl'

        if os.path.exists(pkl_path):
            print("Dataset found, Loading dataset from:", pkl_path)
            with open(pkl_path, 'rb') as f:
                __data = pickle.load(f)

                self.obs = __data["obs"]
                self.pred = __data["pred"]
                self.robs = __data["robs"]
                self.rpred = __data["rpred"]
        else:

            obs = []
            pred = []
            robs = []
            rpred = []

            files = glob.glob(dir + '*.npy')
            files.sort()
            pbar = tqdm(total=len(files))
            for file in files:
                is_LR = True
                if "_RL_" in file:
                    is_LR = False
                pbar.update(1)
                data = np.load(file)
                for t in range(0, data.shape[0] - pred_len - obs_len,
                               moving_window_size):
                    srrounding_index = [0, 1]
                    for k in range(2, data.shape[1]):
                        norm = np.linalg.norm(data[t:t + obs_len, 0, :] -
                                              data[t:t + obs_len, k, :],
                                              ord=None,
                                              axis=-1).mean()
                        if norm < srrounding_limit:
                            srrounding_index.append(k)

                    lvx = data[t, 1, 0]
                    lvy = data[t, 1, 1]
                    evx = data[t + obs_len, 0, 0]
                    evy = data[t + obs_len, 0, 1]

                    #Only care about intersections, avoid linear routes
                    if is_LR and not ((lvy > -4374.36 and evy > -4205.00) or
                                      (lvy > -1074.42 and evx < -2395.39) or
                                      (lvx < -5525.84 and evx < -5695.27) or
                                      (lvx < -8825.73 and evy > -904.95) or
                                      (lvy > 2225.55 and evy > 2395.23)):
                        continue

                    if not is_LR and not (
                        (lvy > -4374.36 and evy > -4205.00) or
                        (lvy > -1074.42 and evx > -8825.60) or
                        (lvx > -5695.27 and evx > -5525.84) or
                        (lvx > -996.06 and evy > -904.95) or
                        (lvy > 2225.55 and evy > 2395.23)):
                        continue

                    _obs = data[t:t + obs_len, srrounding_index].copy()
                    _pred = data[t + obs_len:t + obs_len + pred_len,
                                 0:1].copy()
                    _pred_start = _pred[0,
                                        0:1, :2].copy() - _obs[-1,
                                                               0:1, :2].copy()
                    _obs[1:, :, :2] = _obs[1:, :, :2] - _obs[:-1, :, :2]
                    _obs[0] = 0
                    _pred[1:,
                          0:1, :2] = _pred[1:, 0:1, :2] - _pred[:-1, 0:1, :2]
                    _pred[0, 0:1, :2] = _pred_start
                    robs.append(torch.from_numpy(_obs.astype('float32')))
                    rpred.append(torch.from_numpy(_pred.astype('float32')))
                    obs.append(
                        torch.from_numpy(
                            data[t:t + obs_len,
                                 srrounding_index].astype('float32')))
                    pred.append(
                        torch.from_numpy(
                            data[t + obs_len:t + obs_len + pred_len,
                                 0:1].astype('float32')))
            pbar.close()
            self.obs = obs
            self.pred = pred
            self.robs = robs
            self.rpred = rpred

            __data = {}
            __data["obs"] = self.obs
            __data["pred"] = self.pred
            __data["robs"] = self.robs
            __data["rpred"] = self.rpred
            print("Saving dataset to:", pkl_path)
            with open(pkl_path, "wb") as output_file:
                pickle.dump(__data, output_file)

    def __getitem__(self, idx):

        return self.obs[idx], self.pred[idx], self.robs[idx], self.rpred[idx]

    def __len__(self):
        return len(self.obs)