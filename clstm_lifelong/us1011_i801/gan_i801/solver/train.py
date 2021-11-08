from __future__ import print_function
import torch
from model import highwayNet
from utils import ngsimDataset,maskedNLL,maskedMSE,maskedNLLTest

from model_gan import highwayNet_g_compose
from utils_gan import ngsimDatasetGan
from torch.utils.data import DataLoader
import scipy.io as sio
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import os
plt.switch_backend('agg')

# when a new gan model is trained
# copy train,model,util files here and add a gan postfix
# ngsimDataset in utils_gan also add Gan postfix
# generator arguments copied from train_gan and ignore args_d
# check generator input arguments and dataloader output numbers
# copy raw training and testing data from data/raw to data folder
args_d = {}
args_d['use_cuda'] = True
args_d['in_length'] = 41
args_d['out_length'] = 41
args_d['encoder_size'] = 64
args_d['input_embedding_size_d'] = 32
args_d['batch_size'] = 64
# args_d['traj_num'] = 4

args_g={}
args_g['use_cuda'] = True
args_g['out_length'] = 41
args_g['input_embedding_size_g'] = 64
args_g['encoder_size'] = 128
args_g['batch_size'] = args_d['batch_size']
args_g['latent_dim'] = 16
# args_g['traj_num'] = 4

# args_gd = {}
# args_gd['use_cuda'] = True
# args_gd['in_length'] = 81
# args_gd['out_length'] = 81
# args_gd['encoder_size'] = 512
# args_gd['latent_dim'] = 100
# args_gd['batch_size'] = 128
# args_gd['input_embedding_size_g'] = 128
#
# args_vae = {}
# args_vae['in_length'] = 81*2
# args_vae['input_embedding_size'] = 512
# args_vae['encoder_size'] = 256
# args_vae['z_dim'] = 128
# args_vae['traj_num'] = 4

## Network Arguments
args = {}
args['use_cuda'] = True
args['encoder_size'] = 256
args['decoder_size'] = 256
args['in_length'] = 16
args['out_length'] = 25
args['grid_size'] = (13,3)
args['soc_conv_depth'] = 64
args['conv_3x1_depth'] = 16
args['dyn_embedding_size'] = 64
args['input_embedding_size'] = 64
args['num_lat_classes'] = 3
args['num_lon_classes'] = 2
args['use_maneuvers'] = False
args['train_flag'] = True

# 这里需要注意scale需要以前30步为标准建立，如果以80步建立就会得到很大的偏差
# Initialize network
net = highwayNet(args)
net_g_compose = highwayNet_g_compose(args_g)
# net_g_decompose = highwayNet_g_decompose(args_gd)
# net_vae = VAE(args_vae)

# net_g_decompose.load_state_dict(torch.load('trained_models/ngsim_wcgenerator_decompose_us1011.tar'))
model_name = 'trained_models/generator_i801_nsl_epoch600.tar'
net_g_compose.load_state_dict(torch.load(model_name))
print(model_name,' loaded!')
# net_vae.load_state_dict(torch.load('trained_models/ngsim_vae_decompose_us1011_nei3_epoch483.tar'))
if args['use_cuda']:
    net = net.cuda()
    # net_vae = net_vae.cuda()
    net_g_compose = net_g_compose.cuda()
    # net_g_decompose = net_g_decompose.cuda()

## Initialize optimizer
pretrainEpochs = 15
trainEpochs = 6
optimizer = torch.optim.Adam(net.parameters())
batch_size = args_g['batch_size']
crossEnt = torch.nn.BCELoss()

## Initialize data loaders
real_set_name = '../data/TrainSet-i801-gan-smooth-nei3&4-nsl.mat'
# real_set_name = 'data/TrainSet-carla-1-nei1-maxgrid.mat'
hero_grid_seq = 19
t_short = args_g['out_length']
trSet = ngsimDatasetGan(mat_file=real_set_name, enc_size=256,
                        batch_size=args_g['batch_size'], 
                        local_normalise=1)
trDataloader = DataLoader(trSet,batch_size=args_g['batch_size'],shuffle=True,num_workers=4,
                          drop_last = True, collate_fn=trSet.collate_fn)
enc_size = args['encoder_size']
valSet = ngsimDataset('../data/ValSet-i801-gan-smooth-nei3&4-nsl.mat',grid_size=args['grid_size'], enc_size=enc_size)
#valSet = ngsimDataset('data/ValSet-carla-9-nei1-maxgrid.mat',grid_size=args['grid_size'],
#                      t_h = t_h,t_f = t_f,d_s = d_s,enc_size=enc_size)
valDataloader = DataLoader(valSet,batch_size=batch_size,shuffle=True,num_workers=4,collate_fn=valSet.collate_fn)


def scene2regdata(scene,hero_index,nbr_index,index_division):
    t_h = 15
    t_f = 25
    d_s = 1
    scene = scene.view(-1,2,args_g['out_length'])
    hero_traj = scene[hero_index,:,:]
    ref_pose = hero_traj[:, :, t_h].unsqueeze(2)

    # scene = scene.view(args_g['batch_size'],args_g['traj_num'],2,args_g['out_length'])
    index_batch = np.arange(0, len(index_division))
    len_list = [len(i) for i in index_division]
    index_batch_rep = np.repeat(index_batch, len_list)
    ref_pose = ref_pose[index_batch_rep, :, :]

    scene = scene-ref_pose
    scene_scale = torch.cat([abs(scene[id, :, 0:t_h + 1:d_s]).max(2)[0].max(0)[0].unsqueeze(0).unsqueeze(2)
                             for id in index_division], dim=0)
    scene_scale = scene_scale[index_batch_rep, :]
    # scene_scale = abs(scene[:,:,:,0:t_h+1:d_s]).max(3,keepdim=True)[0].max(1,keepdim=True)[0]
    scene_scaled = scene/scene_scale

    # scene_scaled = scene_scaled.view(-1,2,args_g['out_length'])
    hero_traj = scene_scaled[hero_index, :, :]
    nbr_traj = scene_scaled[nbr_index, :, :]

    # index_division = index_division.detach()
    for id, idx in enumerate(index_division):
        idx = np.array(idx) - id
        idx = np.arange(idx[0], idx[-1])
        index_division[id] = idx.tolist()
        # nbr_traj[idx, :, :] = nbr_traj[idx, :, :] - ref_pose[id, :].unsqueeze(1)
        # hero_traj[id, :, :] = hero_traj[id, :, :] - ref_pose[id, :].unsqueeze(1)
        # this is for generated
        # current_scene = torch.cat((nbr_traj[idx, :, :], hero_traj[id, :, :].unsqueeze(0)), 0)
        # fut_scale = abs(current_scene[:, :, 0:t_h + 1:d_s]).max(2)[0].max(0)[0]
        # nbr_traj[idx, :, :] = nbr_traj[idx, :, :] / fut_scale.unsqueeze(0).unsqueeze(2)
        # hero_traj[id, :, :] = hero_traj[id, :, :] / fut_scale.unsqueeze(0).unsqueeze(2)
    # print(nbr_traj.shape[0]-composed_mask[:,:,:,0].nonzero().shape[0])
    # index_division.requires_grad_(True)
    hist = hero_traj[:, :, 0:t_h + 1:d_s].permute(2, 0, 1)
    fut = hero_traj[:, :, t_h + d_s:t_h + t_f + d_s:d_s].permute(2, 0, 1)
    nbrs = nbr_traj[:, :, 0:t_h + 1:d_s].permute(2, 0, 1)
    return hist,fut,nbrs


## Variables holding train and validation loss values:
train_loss = []
best_val_loss = 99999.9
best_tr_loss = 99999.9
val_loss = []
prev_val_loss = math.inf
draw_num = 100
draw_count = 0
for epoch_num in range(pretrainEpochs):
    if epoch_num == 0:
        print('Pre-training with MSE loss')
    elif epoch_num == pretrainEpochs:
        print('Training with NLL loss')


    ## Train:_________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
    net.train_flag = True

    # Variables to track training performance:
    avg_tr_loss = 0
    total_tr_loss = 0
    avg_tr_time = 0
    avg_lat_acc = 0
    avg_lon_acc = 0

    for i, data in enumerate(trDataloader):

        st_time = time.time()
        real_scene, _,_,mask, index_division,maneuver,scale = data

        if args_g['use_cuda']:
            maneuver = maneuver.cuda()
            mask = mask.cuda()
            mask_pos = mask[:, :, :, 0].nonzero()[:, 1:3].float()
            pos_condition = (mask_pos - torch.tensor([1.0, 6.0]).cuda()) / \
                            (torch.tensor([1.0, 6.0]).cuda())
            nbrs_pos_index = torch.nonzero(torch.sum(abs(pos_condition), dim=1), as_tuple=False)
            nbrs_pos_index = nbrs_pos_index.squeeze(1)
            pos_condition_nbr = pos_condition[nbrs_pos_index, :]
            # pos_condition_nbr = pos_condition_nbr.view(args_d['batch_size'], -1)

            hero_pos_index = torch.where(abs(pos_condition).sum(dim=1) == 0)[0]
            nbrs_pos_index = torch.where(abs(pos_condition).sum(dim=1) > 0)[0]
            # pos_condition = pos_condition.view(args_d['batch_size'], -1)
            mask_traj = mask[:, :, :, 0:2 * args_g['out_length']]
            real_scene = real_scene.cuda()
            # print(scale.shape,real_scene.shape)
            # hero_index = np.arange(scale.shape[0]).tolist()
            # index_len = [len(i) for i in index_division]
            # hero_repeated = np.repeat(hero_index, index_len)
            scale = scale.to(torch.float32).cuda()
            fut_scale = torch.cat([scale[id[0], :].unsqueeze(0) for id in index_division], axis=0)
            real_scene = real_scene/scale
            # real_scene = real_scene.permute(1, 2, 0).contiguous().view(args_d['batch_size'], -1).float()

        # real_scene_scattered = torch.zeros_like(mask_traj).float()
        # real_scene_scattered = real_scene_scattered.masked_scatter_(
        #     mask_traj,real_scene.permute(1,2,0).contiguous().view(-1,2*args_g['out_length']))
        # composed_scene = real_scene_scattered
        # fake_scene = net_g_decompose(maneuver)
        # fake_scene = fake_scene.cuda()
        # composed_scene = net_g_compose(fake_scene, mask,maneuver,index_division)
        # composed_scene = net_g_compose(mask,maneuver,index_division)
        # z = torch.randn(args_g['batch_size'], args_vae['z_dim']).cuda()
        # z = torch.cat((z, pos_condition), dim=1)
        # composed_scene = net_vae.decoder_fc(z)
        # composed_scene = net_vae.decoder(composed_scene)
        composed_scene = net_g_compose(pos_condition,index_division)
        # hist, fut, nbrs = scene2regdata(real_scene.detach(), hero_pos_index, nbrs_pos_index, index_division)
        hist, fut, nbrs = scene2regdata(composed_scene.detach(), hero_pos_index, nbrs_pos_index, index_division)

        # plot
        chose_one = np.random.randint(0,args_g['batch_size'])
        if draw_count<draw_num:
            plt.plot(hist[:, chose_one, 0].cpu().numpy(), hist[:, chose_one, 1].cpu().numpy(), color='r')
            plt.plot(fut[:, chose_one, 0].cpu().numpy(), fut[:, chose_one, 1].cpu().numpy(), color='b')
            neighbors = nbrs[:,index_division[chose_one],:].cpu().numpy()
            for j in range(int(neighbors.shape[1])):
                plt.plot(neighbors[:,j,0], neighbors[:,j,1],color='r')
            path = 'gen_samples/'
            fullname = np.char.add(path, str(draw_count))
            plt.savefig(fullname.item())
            # plt.show()
            plt.close()
            draw_count += 1

        if args['use_cuda']:
            hist = hist.cuda()
            nbrs = nbrs.cuda()
            mask = mask.cuda()
            lat_enc = torch.zeros(hist.shape[1],2).cuda()
            lon_enc = torch.zeros(hist.shape[1],2).cuda()
            fut = fut.cuda()
            op_mask = torch.ones_like(fut).cuda()
            scale = fut_scale.detach().cpu()
            scale = scale.cuda()

        # Forward pass
        if args['use_maneuvers']:
            # print('hist size: ', hist.size())
            # print('nbrs size: ', nbrs.size())
            fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc,index_division)
            # Pre-train with MSE loss to speed up training
            if epoch_num < pretrainEpochs:
                l = maskedMSE(fut_pred, fut, op_mask)
            else:
            # Train with NLL loss
                l = maskedNLL(fut_pred, fut, op_mask) + crossEnt(lat_pred, lat_enc) + crossEnt(lon_pred, lon_enc)
                avg_lat_acc += (torch.sum(torch.max(lat_pred.data, 1)[1] == torch.max(lat_enc.data, 1)[1])).item() / lat_enc.size()[0]
                avg_lon_acc += (torch.sum(torch.max(lon_pred.data, 1)[1] == torch.max(lon_enc.data, 1)[1])).item() / lon_enc.size()[0]
        else:
            fut_pred = net(hist, nbrs, lat_enc, lon_enc,index_division)
            # print('real',fut.shape)
            # print('pred',fut_pred.shape)
            # fut = fut*scale
            # fut_pred[:,:,0:2] = fut_pred[:,:,0:2]*scale
            if epoch_num < pretrainEpochs:
                l = maskedMSE(fut_pred, fut, op_mask)
            else:
                l = maskedNLL(fut_pred, fut, op_mask)

        # Backprop and update weights
        # print(fut.size())
        # print(fut_pred.size())
        optimizer.zero_grad()
        l.backward()
        a = torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        optimizer.step()

        # Track average train loss and average train time:
        batch_time = time.time()-st_time
        avg_tr_loss += l.item()
        total_tr_loss += l.item()
        avg_tr_time += batch_time

        if i%100 == 99:
            eta = avg_tr_time/100*(len(trSet)/batch_size-i)
            print("Epoch no:",epoch_num+1,"| Epoch progress(%):",format(i/(len(trSet)/batch_size)*100,'0.2f'), "| Avg train loss:",format(avg_tr_loss/100,'0.4f'),"| Acc:",format(avg_lat_acc,'0.4f'),format(avg_lon_acc,'0.4f'), "| Validation loss prev epoch",format(prev_val_loss,'0.4f'), "| ETA(s):",int(eta))
            train_loss.append(avg_tr_loss/100)
            avg_tr_loss = 0
            avg_lat_acc = 0
            avg_lon_acc = 0
            avg_tr_time = 0
    # _________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
    print("current loss: ", total_tr_loss, ' history best loss: ', best_tr_loss)
    if total_tr_loss < best_tr_loss or epoch_num == 0:
        best_tr_loss = total_tr_loss
        torch.save(net.state_dict(), 'trained_models/cslstm_ngsim_nei4_besttr.tar')
        print("Best tr loss updated! current best tr loss is at epoch: ", epoch_num + 1)
    sio.savemat('train_data.mat',{'real':fut.detach().cpu().numpy(),'pred':fut_pred.detach().cpu().numpy()})

    ## Validate:______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
    net.train_flag = False

    print("Epoch",epoch_num+1,'complete. Calculating validation loss...')
    avg_val_loss = 0
    avg_val_lat_acc = 0
    avg_val_lon_acc = 0
    val_batch_count = 0
    total_points = 0

    for i, data in enumerate(valDataloader):
        st_time = time.time()
        hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask,scale,index_division = data
        # print(hist.shape)
        # print(nbrs.shape)
        # print(mask.nonzero().shape)

        if args['use_cuda']:
            hist = hist.cuda()
            nbrs = nbrs.cuda()
            mask = mask.cuda()
            lat_enc = lat_enc.cuda()
            lon_enc = lon_enc.cuda()
            fut = fut.cuda()
            op_mask = op_mask.cuda()
            scale = scale.cuda()

        # Forward pass
        if args['use_maneuvers']:
            if epoch_num < pretrainEpochs:
                # During pre-training with MSE loss, validate with MSE for true maneuver class trajectory
                net.train_flag = True
                fut_pred, _ , _ = net(hist, nbrs, lat_enc, lon_enc,index_division)
                l = maskedMSE(fut_pred, fut, op_mask)
            else:
                # During training with NLL loss, validate with NLL over multi-modal distribution
                fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc,index_division)
                l = maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask,avg_along_time = True)
                avg_val_lat_acc += (torch.sum(torch.max(lat_pred.data, 1)[1] == torch.max(lat_enc.data, 1)[1])).item() / lat_enc.size()[0]
                avg_val_lon_acc += (torch.sum(torch.max(lon_pred.data, 1)[1] == torch.max(lon_enc.data, 1)[1])).item() / lon_enc.size()[0]
        else:
            fut_pred = net(hist, nbrs, lat_enc, lon_enc,index_division)
            fut = fut*scale
            fut_pred[:,:,0:2] = fut_pred[:,:,0:2]*scale
            if epoch_num < pretrainEpochs:
                l = maskedMSE(fut_pred, fut, op_mask)
            else:
                l = maskedNLL(fut_pred, fut, op_mask)

        avg_val_loss += l.item()
        val_batch_count += 1

    sio.savemat('val_data.mat',{'real':fut.detach().cpu().numpy(),'pred':fut_pred.detach().cpu().numpy()})

    # Print validation loss and update display variables
    print('Validation loss :',format(avg_val_loss/val_batch_count,'0.4f'),"| Val Acc:",format(avg_val_lat_acc/val_batch_count*100,'0.4f'),format(avg_val_lon_acc/val_batch_count*100,'0.4f'))
    val_loss.append(avg_val_loss/val_batch_count)
    prev_val_loss = avg_val_loss/val_batch_count
    print("current loss: ", avg_val_loss/val_batch_count, ' history best loss: ', best_val_loss/val_batch_count)
    if avg_val_loss < best_val_loss or epoch_num == 0:
        best_val_loss = avg_val_loss
        torch.save(net.state_dict(), 'trained_models/cslstm_i801_nei3&4_nsl_bestval_sgan600.tar')
        print("Best val loss updated! current best val loss is at epoch: ", epoch_num + 1)
    #__________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________




