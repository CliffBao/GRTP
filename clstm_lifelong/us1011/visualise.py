from __future__ import print_function
import torch
from model_wcgan_decompose import highwayNet_g_compose
from utils_wcgan_decompose import ngsimDataset
from scipy.signal import savgol_filter
from torch.utils.data import DataLoader
import scipy.io as sio
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import os
plt.switch_backend('agg')
plt.rcParams['agg.path.chunksize'] = 10000
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

## Network Arguments
args_d = {}
args_d['use_cuda'] = True
args_d['in_length'] = 81
args_d['out_length'] = 81
args_d['encoder_size'] = 256
args_d['input_embedding_size_d'] = 64
args_d['batch_size'] = 128
args_d['traj_num'] = 4

args_g={}
args_g['use_cuda'] = True
args_g['out_length'] = 81
args_g['input_embedding_size_g'] = 64
args_g['encoder_size'] = 128
args_g['batch_size'] = args_d['batch_size']
args_g['latent_dim'] = 16
args_g['traj_num'] = 4
# Initialize network
real_set_name = 'data/TrainSet-us1011-gan-smooth-nei3.mat'

trSet = ngsimDataset(mat_file=real_set_name, t_f=args_d['in_length'],neigh_num=args_d['traj_num']-1,
                     batch_size=args_d['batch_size'], length = args_d['out_length'],
                     local_normalise=1)

trDataloader = DataLoader(trSet,batch_size=args_d['batch_size'],shuffle=True,num_workers=4,
                          drop_last = True, collate_fn=trSet.collate_fn)

## Variables holding train and validation loss values:

prev_val_loss = math.inf

net_g = highwayNet_g_compose(args_g)
net_g = net_g.cuda()
net_g.load_state_dict(torch.load('trained_models/ngsim_generator_decompose_us1011_nei3_epoch390.tar'))


# we implement WGAN-GP instead
fake_sample = []
hero_index = []
## Variables holding train and validation loss values:
for i, data in enumerate(trDataloader):
    real_scene,_,_,mask,index_division,maneuver,scale = data
    mask = mask.cuda()
    traj_condition = real_scene[0,:,:]
    traj_condition = traj_condition.view(args_d['batch_size'],-1).float().cuda()
    maneuver = maneuver.cuda()
    mask_pos = mask[:, :, :, 0].nonzero()[:, 1:3].float()
    pos_condition = (mask_pos - torch.tensor([1.0, 6.0]).cuda()) / \
                    (torch.tensor([1.0, 6.0]).cuda())
    maneuver = torch.cat((maneuver, pos_condition), dim=1)
    nbrs_pos_index = torch.nonzero(torch.sum(abs(pos_condition), dim=1), as_tuple=False)
    nbrs_pos_index = nbrs_pos_index.squeeze(1)
    hero_pos_index = torch.where(torch.sum(abs(pos_condition), dim=1) == 0)[0]
    pos_condition_nbr = pos_condition[nbrs_pos_index, :]
    pos_condition_nbr = pos_condition_nbr.view(args_d['batch_size'], -1)
    hero_pos_index = torch.where(abs(pos_condition).sum(dim=1) == 0)[0]
    hero_pos_index = [x.cpu().numpy() % args_d['traj_num'] for x in hero_pos_index]
    pos_condition = pos_condition.view(args_d['batch_size'], -1)

    composed_scene = net_g(pos_condition,index_division)
    fake_sample.append(composed_scene)
    hero_index.append(hero_pos_index)
    if len(hero_index)>2:
        break

fake_sample = torch.cat(fake_sample,dim=0)
# hero_index = torch.cat(hero_index)
sio.savemat('fake_trajs.mat',{'traj':fake_sample.detach().cpu().numpy(),'hero_index':np.array(hero_index)})
