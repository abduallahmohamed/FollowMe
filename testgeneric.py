import torch
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from metrics import *
from dataset import VFDataSet
from model import FollowMeSTGCNN
import glob
import argparse
import sys

torch.set_printoptions(sci_mode=False)
import seaborn as sns

parser = argparse.ArgumentParser()

#Model specific parameters
parser.add_argument('--tag',
                    default='test',
                    help='personal tag for the model ')
args = parser.parse_args()


def relative_to_abs(rel_traj, start_pos):
    """
    Inputs:
    - rel_traj: pytorch tensor of shape (batch, seq_len, 2)
    - start_pos: pytorch tensor of shape (batch, 2)
    Outputs:
    - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
    """
    # batch, seq_len, 2
    # rel_traj = rel_traj.permute(1, 0, 2)
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj  #.permute(1, 0, 2)


def eval(K=20):
    model.eval()
    pbar = tqdm(total=len(loader_test), desc='Testing:')
    all_pred = []
    all_trgt = []
    all_start = []
    all_obs = []
    with torch.no_grad():
        for cnt, batch in enumerate(loader_test):
            pbar.update(1)
            #Get data
            obs, pred, robs, rpred = batch
            robs = robs.cuda()
            #Forward
            pred_bag = []
            for k in range(K):
                future = model(robs)
                # print(future.shape)
                if len(future.shape) < 4:
                    pred_bag.append(
                        relative_to_abs(future.detach().cpu(), obs[:, -1,
                                                                   0, :2]))
                else:
                    pred_bag.append(
                        relative_to_abs(
                            future[:, :, 0:1, :2].squeeze(2).detach().cpu(),
                            obs[:, -1, 0, :2]))
            pred_bag = torch.cat(pred_bag, dim=0)
            # print(pred_bag.shape)
            # break
            all_pred.append(pred_bag.detach().cpu().clone())
            all_trgt.append(
                pred.squeeze(0).squeeze(1)[..., :2].detach().cpu().clone())
            all_start.append(obs[:, -1,
                                 0, :2].repeat(K, 1).detach().cpu().clone())
            all_obs.append(obs[..., :2].detach().cpu().clone())
            # if cnt == 31:
            #     break

    all_pred_abs = torch.stack(all_pred)
    all_trgt = torch.stack(all_trgt)
    # all_start = torch.stack(all_start)
    # all_obs = torch.stack(all_obs)
    # print(all_pred)
    # b, s, t, l = all_pred.shape
    # all_pred_batch = all_pred.reshape(b * s, t, l)
    # all_start_batch = all_start.reshape(b * s, l)
    # all_pred_abs = relative_to_abs(all_pred_batch,
    #                                all_start_batch).reshape(b, s, t, l)

    pbar.close()

    return all_trgt, all_pred_abs, all_obs, all_start, all_pred


chkpnts = glob.glob('checkpoint/*' + args.tag + '*/')
for checkpoint in chkpnts:
    try:
        with open(checkpoint + 'args.pkl', 'rb') as f:
            args = pickle.load(f)

        dset_test = VFDataSet('processed/test/',
                              obs_len=args.obs_time,
                              pred_len=args.pred_time,
                              srrounding_limit=args.srrounding_limit,
                              moving_window_size=args.moving_window_size)
        loader_test = DataLoader(
            dset_test,
            batch_size=1,  #This is irrelative to the args batch size parameter
            shuffle=False,
            num_workers=0)

        with open(checkpoint + 'metrics.pkl', 'rb') as f:
            metrics = pickle.load(f)
        with open(checkpoint + 'constant_metrics.pkl', 'rb') as f:
            constant_metrics = pickle.load(f)
        with open(checkpoint + 'loss_store.pkl', 'rb') as f:
            loss_store = pickle.load(f)

        model = FollowMeSTGCNN(obs_time=args.obs_time,
                               pred_time=args.pred_time,
                               full_dim=args.full_dim,
                               f_in=args.f_in,
                               f_out=args.f_out).cuda()
        model.load_state_dict(torch.load(checkpoint + 'val_best.pth'))

        all_trgt, all_pred_abs, _, _, _ = eval(K=2)
        ade = min_ade(all_trgt, all_pred_abs)
        fde = min_fde(all_trgt, all_pred_abs)

        f = open("results/" + args.tag + '.txt', "w")
        f.write("Results of:" + checkpoint + "\n")
        f.write("ADE/FDE: " + str(ade) + " | " + str(fde) + "\n")

        print("Results of:", checkpoint)
        print("ADE/FDE: ", ade, fde)

        all_trgt, all_pred_abs, _, _, _ = eval(K=2)

        AMD, AMV = calc_amd_amv(all_trgt, all_pred_abs)
        print("AMD/AMV: ", AMD, AMV)
        f.write("AMD/AMV: " + str(AMD) + " | " + str(AMV) + "\n")

        KDE = kde_lossf(all_trgt, all_pred_abs)
        print("KDE: ", KDE)
        f.write("KDE: " + str(KDE) + "\n")
        f.close()

    except:
        continue
