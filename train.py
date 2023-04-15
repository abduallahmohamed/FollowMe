import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import VFDataSet
from metrics import *
import pickle
import argparse
from model import FollowMeSTGCNN
from tqdm import tqdm

parser = argparse.ArgumentParser()

#Model specific parameters
parser.add_argument('--obs_time', type=int, default=10)
parser.add_argument('--pred_time', type=int, default=30)
parser.add_argument('--full_dim', type=int, default=5)
parser.add_argument('--f_in', type=int, default=2)
parser.add_argument('--f_out', type=int, default=2)

#Loss specific parameters
parser.add_argument('--w_trip', type=float, default=0.0001)

#Dataset specific parameters
parser.add_argument('--srrounding_limit', type=int, default=1000)
parser.add_argument('--moving_window_size', type=int, default=10)

#Training specifc parameters
parser.add_argument('--store_per', type=float, default=0.03)
parser.add_argument('--batch_size',
                    type=int,
                    default=128,
                    help='minibatch size')
parser.add_argument('--num_epochs',
                    type=int,
                    default=120,
                    help='number of epochs')
parser.add_argument('--clip_grad',
                    type=float,
                    default=None,
                    help='gadient clipping')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--lr_sh_rate',
                    type=int,
                    default=40,
                    help='number of steps to drop the lr')
parser.add_argument('--log_freq', type=int, default=2000, help='log frequency')
parser.add_argument('--tag',
                    default='testM3_1',
                    help='personal tag for the model ')
args = parser.parse_args()

print('*' * 30)
print("Training initiating....")
print(args)

#Dataset
dset_train = VFDataSet('processed/train/',
                       obs_len=args.obs_time,
                       pred_len=args.pred_time,
                       srrounding_limit=args.srrounding_limit,
                       moving_window_size=args.moving_window_size)
loader_train = DataLoader(
    dset_train,
    batch_size=1,  #This is irrelative to the args batch size parameter
    shuffle=True,
    num_workers=0)

dset_val = VFDataSet('processed/val/',
                     obs_len=args.obs_time,
                     pred_len=args.pred_time,
                     srrounding_limit=args.srrounding_limit,
                     moving_window_size=args.moving_window_size)
loader_val = DataLoader(
    dset_val,
    batch_size=1,  #This is irrelative to the args batch size parameter
    shuffle=False,
    num_workers=1)

#Loss function
loss_store = {"traj_loss": 0, "trip_loss": 0, "total": 0}
generic_loss_store = {
    'train': {
        "traj_loss": [],
        "trip_loss": [],
        "total": []
    },
    'val': {
        "traj_loss": [],
        "trip_loss": [],
        "total": []
    }
}


def reset_loss_store(isVal=False):
    global generic_loss_store, loss_store
    key = 'train'
    if isVal:
        key = 'val'
    for k, v in loss_store.items():
        generic_loss_store[key][k].append(v)
        loss_store[k] = 0


def div_loss_store(dvd_by):
    global loss_store
    for k, v in loss_store.items():
        loss_store[k] = v / dvd_by


_l1_mean = nn.L1Loss()
_mse_mean = nn.MSELoss()


def trajectory_loss(pred, target):
    # print(pred.shape, target.shape)

    error = ((pred - target)**2).sum(dim=(-1, -2))

    _, indices = torch.sort(error)

    trip_loss = _mse_mean(pred[indices[0]], pred[indices[1]]) - _mse_mean(
        pred[indices[0]], pred[indices[-1]])

    min_error = error.min()
    traj_loss = min_error / (args.pred_time * 2)  #2 = location dimension
    loss_store["traj_loss"] += traj_loss.item()
    loss_store["trip_loss"] += trip_loss.item()

    total_loss = traj_loss + args.w_trip * trip_loss
    loss_store["total"] += total_loss.item()

    return total_loss


def trajectory_loss_val(pred, target):

    error = ((pred - target)**2).sum(dim=(-1, -2))

    min_error = error.min()
    traj_loss = min_error / (args.pred_time * 2)  #2 = location dimension

    total_loss = traj_loss
    loss_store["total"] += total_loss.item()

    return total_loss


#Model

model = FollowMeSTGCNN(obs_time=args.obs_time,
                       pred_time=args.pred_time,
                       full_dim=args.full_dim,
                       f_in=args.f_in,
                       f_out=args.f_out).cuda()

#Optimizer and Schedule
optimizer = optim.SGD(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer,
                                      step_size=args.lr_sh_rate,
                                      gamma=0.1)

#Check pointing
checkpoint_dir = './checkpoint/' + args.tag + '/'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

with open(checkpoint_dir + 'args.pkl', 'wb') as fp:
    pickle.dump(args, fp)

print('Data and model loaded')
print('Checkpoint dir:', checkpoint_dir)

#Training
metrics = {'train_loss': [], 'val_loss': []}
constant_metrics = {'min_val_epoch': -1, 'min_val_loss': 9999999999999999}


def train(epoch):
    global metrics, loader_train, loss_store
    model.train()

    total_loss = 0
    batch_loss = 0
    learn_cnt = 0
    pbar = tqdm(total=len(loader_train), desc='Training:')
    print('Training..')
    for cnt, batch in enumerate(loader_train):
        #Get data
        obs, pred, robs, rpred = batch
        robs, rpred = robs.cuda(), rpred.cuda()

        optimizer.zero_grad()

        #Forward
        futures = []
        for k in range(20):
            future = model(robs)
            # print(future.shape)# 1,30,2
            futures.append(future)
        future = torch.cat(futures, dim=0)
        # future = future[:, :, 0:1, :].squeeze(2)
        # print(rpred.shape, future.shape)
        # print(rpred.squeeze(2)[..., :2].shape, "<<<<<")  #1, 30,2
        #Loss
        batch_loss += trajectory_loss(future, rpred.squeeze(2)[..., :2])

        #Learn
        if cnt % args.batch_size == 0 and cnt != 0:
            batch_loss = batch_loss / args.batch_size
            batch_loss.backward()
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               args.clip_grad)
            optimizer.step()
            #Reset
            learn_cnt += 1
            total_loss += batch_loss.item()
            batch_loss = 0
        #Log
        if cnt % args.log_freq == 0 and cnt != 0:
            # print(args.tag, ' |TRAIN:', '\t Epoch:', epoch, '\t Batch loss:',
            #       batch_loss.item())
            div_loss_store(args.log_freq)
            # print("Detailed train loss:", loss_store)
            pbar.set_postfix(loss_store)
            reset_loss_store(False)
            if epoch > 2:
                vald()

        pbar.update(1)

    metrics['train_loss'].append(total_loss / learn_cnt)
    pbar.close()


iteration = 0


def vald():
    global metrics, loader_val, constant_metrics, iteration, loss_store
    model.eval()
    total_loss = 0

    pbar = tqdm(total=len(loader_val))
    print("Validating...")
    with torch.no_grad():
        for cnt, batch in enumerate(loader_val):
            pbar.update(1)
            #Get data
            obs, pred, robs, rpred = batch
            robs, rpred = robs.cuda(), rpred.cuda()
            #Forward
            future = model(robs)
            # future = future[:, :, 0:1, :].squeeze(2)

            #Loss
            total_loss += trajectory_loss_val(
                future,
                rpred.squeeze(2)[..., :2]).item()

        print(args.tag, ' |VALD:', '\t Iteration:', iteration, '\t Loss:',
              total_loss / (cnt + 1))
        metrics['val_loss'].append(total_loss / (cnt + 1))

        div_loss_store(cnt + 1)
        print("Detailed val loss:", loss_store)
        reset_loss_store(True)
        store_per = args.store_per * constant_metrics['min_val_loss']

        if (constant_metrics['min_val_loss'] -
                metrics['val_loss'][-1]) > store_per:
            # if metrics['val_loss'][-1] < constant_metrics['min_val_loss']:
            constant_metrics['min_val_loss'] = metrics['val_loss'][-1]
            constant_metrics['min_val_epoch'] = iteration
            torch.save(model.state_dict(),
                       checkpoint_dir + 'val_best.pth')  # OK
    iteration += 1
    pbar.close()


print('Training started ...')
for epoch in range(args.num_epochs):
    train(epoch)
    scheduler.step()

    print('*' * 30)
    print(args.tag, ' |Epoch:', args.tag, ":", epoch)
    for k, v in metrics.items():
        if len(v) > 0:
            print(k, v[-1])

    print(constant_metrics)
    print('*' * 30)

    with open(checkpoint_dir + 'metrics.pkl', 'wb') as fp:
        pickle.dump(metrics, fp)

    with open(checkpoint_dir + 'constant_metrics.pkl', 'wb') as fp:
        pickle.dump(constant_metrics, fp)

    with open(checkpoint_dir + 'loss_store.pkl', 'wb') as fp:
        pickle.dump(generic_loss_store, fp)