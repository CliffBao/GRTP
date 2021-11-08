from __future__ import print_function
import torch
from model_wcgan_decompose import highwayNet_d, highwayNet_g_compose, highwayNet
from utils_wcgan_decompose import ngsimDataset,maskedNLL,maskedMSE,maskedNLLTest,mmd_mine
from scipy.signal import savgol_filter
from torch.utils.data import DataLoader
import scipy.io as sio
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import itertools
plt.switch_backend('agg')

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

## Network Arguments
args_d = {}
args_d['use_cuda'] = True
args_d['in_length'] = 41
args_d['out_length'] = 41
args_d['encoder_size'] = 256
args_d['input_embedding_size_d'] = 64
args_d['batch_size'] = 512
args_d['class_num'] = 3

args_g={}
args_g['use_cuda'] = True
args_g['out_length'] = 41
args_g['input_embedding_size_g'] = 64
args_g['encoder_size'] = 128
args_g['batch_size'] = args_d['batch_size']
args_g['latent_dim'] = 16

args_g_trained={}
args_g_trained['use_cuda'] = True
args_g_trained['out_length'] = 41
args_g_trained['input_embedding_size_g'] = 64
args_g_trained['encoder_size'] = 128
args_g_trained['batch_size'] = args_d['batch_size']//2
args_g_trained['latent_dim'] = 16

args = {}
args['use_cuda'] = True
args['encoder_size'] = 256
args['decoder_size'] = 256
args['in_length'] = 16
args['out_length'] = 25
args['all_length'] = 41
args['batch_size'] = args_d['batch_size']
args['grid_size'] = (13,3)
args['soc_conv_depth'] = 64
args['conv_3x1_depth'] = 16
args['dyn_embedding_size'] = 64
args['input_embedding_size'] = 64
args['num_lat_classes'] = 3
args['num_lon_classes'] = 2
args['use_maneuvers'] = False
args['train_flag'] = True
# nei 3
# args['class_num'] = 10
# nei 4
# args['class_num'] = 15
# nei3 & nei4
args['class_num'] = 25
# Initialize network

## Initialize optimizer
# train generator every n times
trainEpochs = 3000
lambda_gp = 10
n_critic = 4
learning_rate_d = 0.0001
learning_rate_g = 0.0001
b1 = 0.90
b2 = 0.999
# clip_value = 0.01
d_loss_record = []
d_fake_loss_record = []
d_real_loss_record = []
d_penalty_loss_record = []
g_loss_record = []
## Variables holding train and validation loss values:
train_loss = []
val_loss = []

prev_val_loss = math.inf

# torch.autograd.set_detect_anomaly(True)

def compute_gradient_penalty(D, real_samples, fake_samples,index):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    # sample shape:(time_length, batch_size*scene_traj_num, dim) = (81, 128, 2)
    real_samples = real_samples.permute(1,2,0)
    fake_samples = fake_samples.permute(1,2,0)
    # (batch_size*scene_traj_num,2,time_length)
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

    # scene_vel = diff(interpolates, 1)
    # scene_acc = diff(interpolates, 2)
    # interpolates = torch.cat((interpolates, scene_vel, scene_acc), 1)
    interpolates = interpolates.permute(2, 0, 1)

    d_interpolates = D(interpolates,index)
    out_shape = real_samples.size(0)
    fake = Variable(Tensor(out_shape,1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    # with torch.autograd.detect_anomaly():
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.contiguous().view(gradients.size(0), -1)
    # b = torch.nn.utils.clip_grad_norm_(gradients, 10)
    # GP
    # gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    # LP
    # gp = (torch.clamp(gradients.norm(2, dim=1) - 1, 0, float("inf")) ** 2).mean()
    # wgan-div
    gp = (gradients.norm(2, dim=1) ** 6).mean()
    return gp


def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg


def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def log_sum_exp(x, axis=1):
    m = torch.max(x, dim=1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(1)), dim=axis))


def scene2regdata(scene,hero_index,nbr_index,index_division):
    t_h = 15
    t_f = 25
    d_s = 1
    scene = scene.view(-1,2,args_g['out_length'])
    hero_traj = scene[hero_index,:,:]
    ref_pose = hero_traj[:, :, t_h]

    scene = scene.view(args_g['batch_size'],args_g['traj_num'],2,args_g['out_length'])
    scene = scene-ref_pose.unsqueeze(1).unsqueeze(3)
    scene_scale = abs(scene[:,:,:,0:t_h+1:d_s]).max(3)[0].max(1)[0]
    scene_scaled = scene/scene_scale.unsqueeze(1).unsqueeze(3)

    scene_scaled = scene_scaled.view(-1,2,args_g['out_length'])
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
    fut = hero_traj[:, :, t_h + d_s:t_h + t_f + 1:d_s].permute(2, 0, 1)
    nbrs = nbr_traj[:, :, 0:t_h + 1:d_s].permute(2, 0, 1)
    return hist,fut,nbrs

sample_interval = 1000
image_id = 0
# we first generate single trajectory dataset, then we load it to speed up compose training
# net_g = highwayNet_g_single(args_g)
# if args_g['use_cuda']:
#     net_g = net_g.cuda()
# net_g.load_state_dict(torch.load('trained_models/ngsim_wcgenerator_single_selu_full_d1000epoch_cond.tar'))
# for first run, set gen_set_name = None
# gen_set_name = 'data/Trainset_ngsim_clstm_smooth_wcgan_single_manuever6_fake.mat'
# gen_set_name = None
# args_g['neigh_num'] = 6
real_set_name = 'data/TrainSet-i801-gan-smooth-nei3&4.mat'

trSet = ngsimDataset(mat_file=real_set_name,batch_size=args_d['batch_size']//2,local_normalise=1)

trDataloader = DataLoader(trSet,batch_size=args_d['batch_size']//2,shuffle=True,num_workers=4,
                          drop_last = True, collate_fn=trSet.collate_fn)

net_d = highwayNet_d(args_d)
net_g = highwayNet_g_compose(args_g)
net = highwayNet(args)
net_g_us1011 = highwayNet_g_compose(args_g_trained)
# net_g_i801 = highwayNet_g_compose(args_g_trained)

# we can start from last saved models
net_g_us1011.load_state_dict(torch.load('trained_models/generator_us1011_nsl_epoch201.tar'))
# net_g_i801.load_state_dict(torch.load('trained_models/generator_i801_nsl_epoch600.tar'))

if args_d['use_cuda']:
    net_d = net_d.cuda()
    net_g = net_g.cuda()
    net = net.cuda()
    net_g_us1011 = net_g_us1011.cuda()
    # net_g_i801 = net_g_i801.cuda()

# clip_value = 0.01
d_loss_record = []
g_loss_record = []
reg_loss_record = []
g_loss_reg_record = []
d_fake_loss_record = []
d_real_loss_record = []
d_penalty_loss_record = []
# we implement WGAN-GP instead
# optimizer_d = torch.optim.RMSprop(net_d_compose.parameters(), lr = learning_rate_d)
# optimizer_g = torch.optim.RMSprop(net_g_compose.parameters(), lr = learning_rate_g)
optimizer_d = torch.optim.Adam(net_d.parameters(), lr = learning_rate_d, betas = (b1,b2))
optimizer_g = torch.optim.Adam(net_g.parameters(), lr = learning_rate_g, betas = (b1,b2))
optimizer = torch.optim.Adam(net.parameters(),lr = learning_rate_g, betas = (b1,b2))

Tensor = torch.cuda.FloatTensor if args_d['use_cuda'] else torch.FloatTensor
criterion_dae = torch.nn.MSELoss()
batch_size = args_d['batch_size']

## Variables holding train and validation loss values:
train_loss = []
val_loss = []
prev_val_loss = math.inf

image_id = 0
mmd_record = np.zeros(trainEpochs)
mmd_epoch = np.zeros((trSet.__len__()//args_d['batch_size']))
mean_real = []
mean_fake = []
# checkpoint part
start_epoch = 0
resume = True
if resume:
    if os.path.isfile('checkpoint.pkl'):
        checkpoint = torch.load('checkpoint.pkl')
        start_epoch = checkpoint['epoch'] + 1
        mmd_record = checkpoint['mmd_record']
        net_d.load_state_dict(checkpoint['model_discriminator'])
        net_g.load_state_dict(checkpoint['model_generator'])
        net.load_state_dict(checkpoint['model_regression'])
        optimizer_d.load_state_dict(checkpoint['optimizer_d'])
        optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        optimizer.load_state_dict(checkpoint['optimizer_regression'])
        g_loss_record = checkpoint['g_loss']
        d_loss_record = checkpoint['d_loss']
        reg_loss_record = checkpoint['reg_loss']
        g_loss_reg_record = checkpoint['g_loss_reg']
        mean_real = checkpoint['mean_real']
        mean_fake = checkpoint['mean_fake']
        d_fake_loss_record = checkpoint['d_fake_loss']
        d_real_loss_record = checkpoint['d_real_loss']
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    else:
        print("=> no checkpoint found")

change_count = 0
# checkpoint part end
y = Variable(Tensor(args_d['batch_size'], 1).fill_(1.0))
t_clipped = args_d['in_length']
regression_epoch = 10
# 懒得写hash映射函数了，直接写个表吧
# nei3 case: 10 classes
nbr_label_2_scene_label_table = torch.zeros(5,5,5)
# nei3 only
base_3 = 0
nbr_label_2_scene_label_table[0,0,3] = 0+base_3
nbr_label_2_scene_label_table[0,1,2] = 1+base_3
nbr_label_2_scene_label_table[0,2,1] = 2+base_3
nbr_label_2_scene_label_table[0,3,0] = 3+base_3
nbr_label_2_scene_label_table[1,2,0] = 4+base_3
nbr_label_2_scene_label_table[1,0,2] = 5+base_3
nbr_label_2_scene_label_table[1,1,1] = 6+base_3
nbr_label_2_scene_label_table[2,0,1] = 7+base_3
nbr_label_2_scene_label_table[2,1,0] = 8+base_3
nbr_label_2_scene_label_table[3,0,0] = 9+base_3
# nei4 case:15 classes
# nbr_label_2_scene_label_table_4 = torch.zeros(4,4,4)
# nei3 + nei4 mixed dataset
base_4 = 10
# nei4 only
# base_4 = 0
nbr_label_2_scene_label_table[0,0,4] = 0+base_4
nbr_label_2_scene_label_table[0,1,3] = 1+base_4
nbr_label_2_scene_label_table[0,2,2] = 2+base_4
nbr_label_2_scene_label_table[0,3,1] = 3+base_4
nbr_label_2_scene_label_table[0,4,0] = 4+base_4
nbr_label_2_scene_label_table[1,3,0] = 5+base_4
nbr_label_2_scene_label_table[1,2,1] = 6+base_4
nbr_label_2_scene_label_table[1,1,2] = 7+base_4
nbr_label_2_scene_label_table[1,0,3] = 8+base_4
nbr_label_2_scene_label_table[2,0,2] = 9+base_4
nbr_label_2_scene_label_table[2,1,1] = 10+base_4
nbr_label_2_scene_label_table[2,2,0] = 11+base_4
nbr_label_2_scene_label_table[3,1,0] = 12+base_4
nbr_label_2_scene_label_table[3,0,1] = 13+base_4
nbr_label_2_scene_label_table[4,0,0] = 14+base_4

sgan_loss = torch.nn.BCELoss()
color_arr=['b', 'g', 'r', 'c', 'm', 'y', 'k']
for start_epoch in range(start_epoch,trainEpochs):
    print('training epoch %d-%d'%(start_epoch,trainEpochs))
    epoch_start_t = time.time()
    ## Train:_________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
    net_d.train_flag = True
    net_g.train_flag = True
    # Variables to track training performance:
    avg_tr_loss_vae = 0
    avg_tr_time = 0

    for i, data in enumerate(trDataloader):
        valid = Variable(Tensor(args_d['batch_size'], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(args_d['batch_size'], 1).fill_(0.0), requires_grad=False)
        st_time = time.time()
        real_scene,_,_,mask,index_division,maneuver,scale = data
        index_division_rep = copy.deepcopy(index_division)
        
        index_division_rep = [list(map(lambda x:x+scale.shape[0],k)) for k in index_division]

        index_division_full = index_division+index_division_rep
        index_division_cp_classify_real = copy.deepcopy(index_division_full)
        index_division_cp_classify_fake = copy.deepcopy(index_division_full)
        index_division_cp_reg_real = copy.deepcopy(index_division_full)
        index_division_cp_reg_fake = copy.deepcopy(index_division_full)
        index_division_cp_gen_creal = copy.deepcopy(index_division_full)
        index_division_cp_gen_cfake = copy.deepcopy(index_division_full)
        index_division_cp_gen_rreal = copy.deepcopy(index_division_full)
        index_division_cp_gen_rfake = copy.deepcopy(index_division_full)
        index_division_plot = copy.deepcopy(index_division_full)

        if args_d['use_cuda']:
            real_scene = real_scene[0:t_clipped,:,:].cuda()
            maneuver = maneuver.cuda()
            scale = scale.cuda()
            real_scene = real_scene/scale
            real_scene = real_scene.permute(1,2,0).float()
            mask = mask.cuda()
            mask_traj = mask[:, :, :, 0:2 * args_g['out_length']]
            mask_pos = mask[:, :, :, 0].nonzero()[:, 1:3].float()
            pos_condition = (mask_pos - torch.tensor([1.0, 6.0]).cuda()) / \
                            (torch.tensor([1.0, 6.0]).cuda())
            pos_condition_full = pos_condition.repeat(2,1)
            maneuver = torch.cat((maneuver, pos_condition), dim=1)
            nbrs_pos_index = torch.nonzero(torch.sum(abs(pos_condition_full), dim=1), as_tuple=False)
            nbrs_pos_index = nbrs_pos_index.squeeze(1)
            hero_pos_index = torch.where(torch.sum(abs(pos_condition_full), dim=1) == 0)[0]
            # pos_condition_nbr = pos_condition[nbrs_pos_index, :]
            # pos_condition_nbr = pos_condition_nbr.view(args_d['batch_size'], -1)
            # pos_condition = pos_condition.view(args_d['batch_size'],-1)

        toggle_grad(net_d, True)
        toggle_grad(net_g, False)
        toggle_grad(net, False)
        net_d.zero_grad()
        optimizer_d.zero_grad()
        with torch.no_grad():
            # create a new real samples by merging samples from two generation models
            scene_us1011 = net_g_us1011(pos_condition,index_division)
            # scene_i801 = net_g_i801(pos_condition,index_division)
            real_scene = torch.cat((scene_us1011,real_scene),axis=0)

            composed_scene = net_g(pos_condition_full,index_division_full)
        # print(torch.norm(composed_scene[hero_pos_index,:,0],dim=1))
        # sio.savemat('net_io.mat',{'pos_condition':pos_condition.detach().cpu().numpy(),
        #                     'index_div':index_division,
        #                     'composed_scene':composed_scene.contiguous().view(-1,2*81).detach().cpu().numpy(),
        #                     'real_scene':real_scene.contiguous().view(-1,2*81).detach().cpu().numpy()})
        composed_scene.requires_grad_()
        real_scene.requires_grad_()
        # pos_condition_nbr.requires_grad_()
        # [time_line,traj_id,traj_xy]
        # y_pred,_ = net_d(real_scene,pos_condition,
        #                  hero_pos_index,nbrs_pos_index,index_division_cp_classify_real)
        # y_pred_fake,_ = net_d(composed_scene,pos_condition,
        #                       hero_pos_index,nbrs_pos_index,index_division_cp_classify_fake)
        # label = pos_condition.view(-1, 2)
        # label_pos = label[:,0].long()+1
        #
        # logz_pred, logz_pred_fake = log_sum_exp(y_pred), log_sum_exp(y_pred_fake)  # log ∑e^x_i
        # prob_label = torch.gather(y_pred, 1, label_pos.unsqueeze(1))  # log e^x_label = x_label
        # loss_supervised = -torch.mean(prob_label) + torch.mean(logz_pred)
        # # real_data: log Z/(1+Z), fake_data: log 1/(1+Z)
        # loss_unsupervised = torch.mean(F.softplus(logz_pred))-torch.mean(logz_pred) + \
        #                     torch.mean(F.softplus(logz_pred_fake))
        # d_loss = loss_supervised + loss_unsupervised
        # # gradient_penalty = compute_gradient_penalty(net_d, real_scene.detach(), composed_scene.detach(),index_division)
        # # d_fake_loss = torch.mean(y_pred_fake)
        # # d_real_loss = -torch.mean(y_pred)
        # # d_loss = d_fake_loss+d_real_loss+lambda_gp*gradient_penalty
        # # fake == unsupervised
        # # real == supervised
        # # d_fake_loss = torch.mean(y_pred_fake)
        # # d_real_loss = -torch.mean(y_pred)
        # # d_loss = d_fake_loss + d_real_loss
        # d_loss.backward()
        # optimizer_d.step()

        # if start_epoch < regression_epoch:
        #     hist,fut,nbrs = scene2regdata(real_scene.detach(),hero_pos_index,nbrs_pos_index,index_division_cp)
        #     hist = hist.cuda()
        #     nbrs = nbrs.cuda()
        #     lat_enc = torch.zeros(hist.shape[1], 2).cuda()
        #     lon_enc = torch.zeros(hist.shape[1], 2).cuda()
        #     fut = fut.cuda()
        #     op_mask = torch.ones_like(fut).cuda()
        #     fut_pred = net(hist, nbrs, lat_enc, lon_enc, index_division_cp)
        #     reg_loss = maskedMSE(fut_pred, fut, op_mask)
        #     optimizer.zero_grad()
        #     reg_loss.backward()
        #     optimizer.step()
        #     reg_loss_record.append(reg_loss.item())
        # regression network works as another discriminator
        toggle_grad(net_d, False)
        toggle_grad(net, True)
        toggle_grad(net_g, False)
        optimizer.zero_grad()
        logit_fake, feature_seq_fake = net(composed_scene, hero_pos_index,nbrs_pos_index,index_division_cp_reg_fake)
        logit_real, feature_seq_real = net(real_scene, hero_pos_index,nbrs_pos_index,index_division_cp_reg_real)
        # sio.savemat('rd_encoder_imp.mat',{'pos_condition':pos_condition.detach().cpu().numpy()})
        # print('imp saved!')
        # time.sleep(5.0)

        # label = pos_condition.view(-1, 2)
        # nbr_label = label[nbrs_pos_index,:]
        # # 0,1,2 label pos
        # nbr_label_pos = nbr_label[:, 0].long() + 1
        # label_value = torch.ones(nbr_label.shape[0],3).cuda()
        # label_onehot = torch.zeros(nbr_label.shape[0],3).cuda()
        # # 001,100,010
        # label_onehot.scatter_(1,nbr_label_pos.unsqueeze(1),label_value)
        # # label_onehot = label_onehot.view(args_d['batch_size'],args_d['traj_num']-1,3)
        # label_onehot_sum = torch.cat([torch.sum(label_onehot[id,:],dim=0).unsqueeze(0)
        #                               for id in index_division_cp_reg_real],dim=0)
        # nbr_scene_label_pos = nbr_label_2_scene_label_table[label_onehot_sum[:,0].long(),
        #                                                     label_onehot_sum[:,1].long(),
        #                                                     label_onehot_sum[:,2].long()].long().cuda()
        # logz_pred_reg, logz_pred_fake_reg = log_sum_exp(logit_real), log_sum_exp(logit_fake)  # log ∑e^x_i
        # prob_label_reg = torch.gather(logit_real, 1, nbr_scene_label_pos.unsqueeze(1))  # log e^x_label = x_label
        # loss_supervised_reg = -torch.mean(prob_label_reg) + torch.mean(logz_pred_reg)
        # # real_data: log Z/(1+Z), fake_data: log 1/(1+Z)
        # loss_unsupervised_reg = torch.mean(F.softplus(logz_pred_reg)) - torch.mean(logz_pred_reg) + \
        #                         torch.mean(F.softplus(logz_pred_fake_reg))
        # g_loss_reg = loss_supervised_reg + loss_unsupervised_reg
        g_loss_reg = (sgan_loss(logit_real,valid)+sgan_loss(logit_fake,fake))/2
        g_loss_reg.backward()
        optimizer.step()

        if i%n_critic==0:
            toggle_grad(net_d, False)
            toggle_grad(net, False)
            toggle_grad(net_g, True)
            net_g.zero_grad()
            optimizer_g.zero_grad()
            # pos_condition_nbr.requires_grad_()
            # maneuver.requires_grad_()
            recon_data = net_g(pos_condition_full,index_division_full)
            # y_pred_fake_gen,fake_feature = net_d(recon_data,pos_condition,
            #                                      hero_pos_index,nbrs_pos_index,index_division_cp_gen_cfake)
            # y_pred_gen,real_feature = net_d(real_scene,pos_condition,
            #                                 hero_pos_index,nbrs_pos_index,index_division_cp_gen_creal)
            # fm_loss_1 = torch.mean(torch.abs(fake_feature.mean(dim=0) - real_feature.mean(dim=0)))
            # gan_loss_1 = -torch.mean(F.softplus(log_sum_exp(y_pred_fake_gen)))
            # g_loss_1 = fm_loss_1+gan_loss_1
            # g_loss_1 = gan_loss_1
            # seq_pred_real, real_seq_feature = net(real_scene,
            #                                       hero_pos_index, nbrs_pos_index, index_division_cp_gen_rreal)
            seq_pred_fake, fake_seq_feature = net(recon_data,
                                                  hero_pos_index,nbrs_pos_index,index_division_cp_gen_rfake)
            # fm_loss_2 = torch.mean(torch.abs(fake_seq_feature.mean(dim=0) - real_seq_feature.mean(dim=0)))
            # gan_loss_2 = -torch.mean(F.softplus(log_sum_exp(seq_pred_fake)))
            # g_loss_2 = fm_loss_2 + gan_loss_2
            g_loss_2 = sgan_loss(seq_pred_fake,valid)
            # print(g_loss_1,g_loss_2,pos_condition,y_pred_fake_gen,seq_pred_fake)
            # g_loss = g_loss_1+g_loss_2
            g_loss = g_loss_2

            g_loss.backward()
            optimizer_g.step()

            d_loss_record.append(0)
            g_loss_record.append(g_loss.detach().cpu().numpy())
            g_loss_reg_record.append(g_loss_reg.item())

        avg_tr_loss_vae += g_loss.item()
        batch_time = time.time()-st_time
        avg_tr_time += batch_time

        if i%sample_interval==0:
            # raw fake sample: batch_size*3*13*161
            sample = recon_data.view(-1,2,args_d['out_length']).permute(2,0,1)
            # sample = sample*scale
            # sample = sample.permute(1,2,0).view(args_d['batch_size'],args_d['traj_num'],2,args_d['out_length'])
            sample = sample.permute(1,2,0)
            sample = sample.detach().cpu().numpy()

            plt.ion()
            draw_num = 3
            for draw in range(draw_num):
                current_id = index_division_plot[draw]
                hero_id = hero_pos_index[draw]
                plt.text(sample[hero_id,0,0],sample[hero_id,1,0],'HERO')
                for j in current_id:
                    plt.plot(sample[j,0,:],sample[j,1,:],color=color_arr[(j-current_id[0])%len(color_arr)])
                    plt.scatter(sample[j,0,0],sample[j,1,0],marker='+')
                path = 'gen_samples/full_compose/'
                fullname = path + str(start_epoch)+'-'+str(draw)
                plt.savefig(fullname)
                # plt.show()
                plt.close()

            sample = real_scene.view(-1, 2, args_d['out_length']).permute(2, 0, 1)
            # sample = sample * scale
            # sample = sample.permute(1,2,0).view(args_d['batch_size'],args_d['traj_num'],2,args_d['out_length'])
            sample = sample.permute(1,2,0)
            sample = sample.detach().cpu().numpy()
            # mask_pos = mask_pos.view(args_d['batch_size'],args_d['traj_num'],2).detach().cpu().numpy()

            for draw in range(draw_num):
                current_id = index_division_plot[draw]
                hero_id = hero_pos_index[draw]
                plt.text(sample[hero_id,0,0],sample[hero_id,1,0],'HERO')
                for j in current_id:
                    plt.plot(sample[j, 0, :], sample[j, 1, :],color=color_arr[(j-current_id[0])%len(color_arr)])
                    plt.scatter(sample[j,0,0],sample[j,1,0],marker='+')
                    # plt.text(sample[draw,j,0,0],sample[draw,j,1,0],str(mask_pos[draw,j,:]))
                path = 'real_samples/full_compose/'
                fullname = path + str(start_epoch)+'-'+str(draw)
                plt.savefig(fullname)
                # plt.show()
                plt.close()

            image_id = image_id+sample_interval

        if i % 100 == 99:
            print("Training Epoch no:", start_epoch + 1,
                  " iter: ", i, "/", len(trDataloader),
                  " d_loss:", g_loss_reg.item(),
                  ' g_loss', g_loss.item(), ' avg g loss: ', avg_tr_loss_vae / 100,
                  ' avg train time:', avg_tr_time)
            train_loss.append(avg_tr_loss_vae / 100)
            avg_tr_loss_vae = 0
            avg_tr_time = 0
    epoch_end_t = time.time()
    # mmd_record[start_epoch] = np.mean(mmd_epoch)
    print("epoch no:",start_epoch+1," spent total time is:", epoch_end_t-epoch_start_t)
    sio.savemat('d_mean_real_fake.mat', {'mean_real': np.array(mean_real), 'mean_fake': np.array(mean_fake)})

    len_epoch = len(g_loss_record)//(start_epoch+1)
    plot_epoch_num = 10

    if start_epoch > plot_epoch_num:
        plt.plot(np.arange(len(g_loss_record)-plot_epoch_num*len_epoch, len(g_loss_record)),
                 g_loss_record[-plot_epoch_num*len_epoch:])
    else:
        plt.plot(np.arange(len(g_loss_record)), g_loss_record)
    path = 'gen_samples/g_loss_epoch' + str(start_epoch)
    # fullname = np.char.add(path, str(image_id + draw))
    plt.savefig(path)
    # plt.show()
    plt.close()
    #
    # plt.plot(np.arange(len(reg_loss_record)), reg_loss_record)
    # path = 'gen_samples/full_reg_loss'
    # plt.savefig(path)
    # plt.show()
    # plt.close()

    if start_epoch > plot_epoch_num:
        plt.plot(np.arange(len(g_loss_reg_record)-plot_epoch_num*len_epoch, len(g_loss_reg_record)),
                 g_loss_reg_record[-plot_epoch_num*len_epoch:])
    else:
        plt.plot(np.arange(len(g_loss_reg_record)), g_loss_reg_record)
    path = 'gen_samples/g_loss_reg_epoch' + str(start_epoch)
    # fullname = np.char.add(path, str(image_id + draw))
    plt.savefig(path)
    # plt.show()
    plt.close()

    plt.plot(np.arange(len(g_loss_reg_record)), g_loss_reg_record)
    path = 'gen_samples/full_g_loss_reg'
    plt.savefig(path)
    # plt.show()
    plt.close()

    plt.plot(np.arange(len(g_loss_record)), g_loss_record)
    plt.savefig('gen_samples/full_generator_loss')
    # plt.show()
    plt.close()

    # if start_epoch > plot_epoch_num:
    #     plt.plot(np.arange(len(d_loss_record)-plot_epoch_num*len_epoch, len(d_loss_record)),
    #              d_loss_record[-plot_epoch_num*len_epoch:])
    # else:
    #     plt.plot(np.arange(len(d_loss_record)), d_loss_record)
    # path = 'gen_samples/d_loss_epoch' + str(start_epoch)
    # # fullname = np.char.add(path, str(image_id + draw))
    # plt.savefig(path)
    # plt.show()
    # plt.close()
    #
    # plt.plot(np.arange(len(d_loss_record)), d_loss_record)
    # plt.savefig('gen_samples/full_epoch_discriminator')
    # plt.show()
    # plt.close()

     # checkpoint,save every 3 epochs
    if start_epoch%3==0 and start_epoch<trainEpochs:
        checkpoint = {
            'epoch': start_epoch,
            'model_generator': net_g.state_dict(),
            'model_discriminator': net_d.state_dict(),
            'model_regression': net.state_dict(),
            'optimizer_g': optimizer_g.state_dict(),
            'optimizer_d': optimizer_d.state_dict(),
            'optimizer_regression': optimizer.state_dict(),
            'mmd_record': mmd_record,
            'g_loss': g_loss_record,
            'd_loss': d_loss_record,
            'reg_loss': reg_loss_record,
            'g_loss_reg': g_loss_reg_record,
            'd_fake_loss': d_fake_loss_record,
            'd_real_loss': d_real_loss_record,
            'mean_real': mean_real,
            'mean_fake': mean_fake
        }
        # torch.cuda.empty_cache()
        torch.save(checkpoint, 'checkpoint.pkl')
        torch.save(net_g.state_dict(),
                   'trained_models/generator_us1011_i801_nsl_epoch' + str(start_epoch) + '.tar')
        # torch.save(net_d.state_dict(),
        #            'trained_models/ngsim_discriminator_us1011_nei3_epoch' + str(start_epoch) + '.tar')
        # torch.save(net.state_dict(),
        #            'trained_models/ngsim_regression_us1011_nei3_epoch' + str(start_epoch) + '.tar')
        sio.savemat('loss.mat', {'d_loss': np.array(d_loss_record),
                                 'g_loss': np.array(g_loss_record),
                                 'g_reg_loss': np.array(g_loss_reg_record)})
        print('checkpoint saved')
    #
    # _________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
torch.save(net_g.state_dict(), 'trained_models/generator_us1011_i801_nsl.tar')
torch.save(net_d.state_dict(), 'trained_models/discriminator_us1011_i801_nsl.tar')
torch.save(net.state_dict(), 'trained_models/regression_us1011_i801_nsl.tar')
sio.savemat('wcgan_compose_loss_100.mat',{'d_loss':d_loss_record,
                                          'g_loss':g_loss_record,
                                          'reg_loss':reg_loss_record,
                                          'g_loss_reg':g_loss_reg_record,
                                          'd_fake_loss':d_fake_loss_record,'d_real_loss':d_real_loss_record})
