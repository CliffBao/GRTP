from __future__ import print_function
import torch
from model import highwayNet
from utils import ngsimDataset,maskedMSE,maskedMSETest
from buffer import Buffer
from tqdm import tqdm,trange

from torch.utils.data import DataLoader
import scipy.io as sio
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
plt.switch_backend('agg')

## Network Arguments
args = {}
args['use_cuda'] = True
# args['encoder_size'] = 64
# args['decoder_size'] = 128
args['in_length'] = 16
args['out_length'] = 25
args['grid_size'] = (13,3)
# args['dyn_embedding_size'] = 32
# args['input_embedding_size'] = 32
args['encoder_size'] = 256
args['decoder_size'] = 256
args['dyn_embedding_size'] = 64
args['input_embedding_size'] = 64
args['batch_size'] = 128
# 这里需要注意scale需要以前30步为标准建立，如果以80步建立就会得到很大的偏差
# Initialize network
net = highwayNet(args)

# net_vae.load_state_dict(torch.load('trained_models/ngsim_vae_decompose_us1011_nei3_epoch483.tar'))
if args['use_cuda']:
    net = net.cuda()

## Initialize optimizer
trainEpochs = 15
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

#
# def RMSE(real,pred):
#     err = torch.sqrt(torch.sum(torch.pow(real-pred,2),2))
#     err = torch.mean(err,axis=1)
#     return torch.mean(err,axis=0)
#
#
# def RMSE_time(real,pred):
#     err = torch.sqrt(torch.sum(torch.pow(real-pred,2),2))
#     err = torch.mean(err,dim=1,keepdim=True)
#     return err

# lifelong datasets
train_set_list = [
    '../../data/TrainSet-us1011-gan-smooth-nei3&4-nsl.mat',
    '../../data/TrainSet-i801-gan-smooth-nei3&4-nsl.mat',
    '../../data/TrainSet-highd20-gan-pos-nsl.mat',
    '../../data/TrainSet-inter5d-gan-lane-clustered.mat',
    '../../data/TrainSet-us1012-gan-smooth-nei3&4-nsl.mat',
]
val_set_list = [
    '../../data/ValSet-us1011-gan-smooth-nei3&4-nsl.mat',
    '../../data/ValSet-i801-gan-smooth-nei3&4-nsl.mat',
    '../../data/ValSet-highd20-gan-pos-nsl.mat',
    '../../data/ValSet-inter5d-gan-lane-clustered.mat',
    '../../data/ValSet-us1012-gan-smooth-nei3&4-nsl.mat',
]
test_set_list = [
    '../../data/TestSet-us1011-gan-smooth-nei3&4-nsl.mat',
    '../../data/TestSet-i801-gan-smooth-nei3&4-nsl.mat',
    '../../data/TestSet-highd20-gan-pos-nsl.mat',
    '../../data/TestSet-inter5d-gan-lane-clustered.mat',
    '../../data/TestSet-us1012-gan-smooth-nei3&4-nsl.mat',
]
buffer_size = 1024
buffer = Buffer(args['batch_size']*buffer_size)
task_id = 0
start_epoch = 0
train_loss = []
val_loss = []
test_loss = []
best_val_loss = math.inf
best_val_model_param = None
model_init_param = net.state_dict()

resume = False
if resume:
    if os.path.isfile('checkpoint.pkl'):
        checkpoint = torch.load('checkpoint.pkl')
        start_epoch = checkpoint['epoch'] + 1
        task_id = checkpoint['task_id']
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        train_loss = checkpoint['train_loss']
        best_val_loss = checkpoint['best_val_loss']
        best_val_model_param = checkpoint['best_val_model_param']
        tqdm.write("=> loaded checkpoint (task {},epoch {})".format(task_id,start_epoch))
    else:
        tqdm.write("=> no checkpoint found")

trial_times = 3

for trial in range(trial_times):
    trial_path = 'trial'+str(trial)+'/'
    buffer_tmp_file = trial_path+"data"
    if not os.path.exists(buffer_tmp_file):
        os.makedirs(buffer_tmp_file)

    net.load_state_dict(model_init_param)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    buffer = Buffer(args['batch_size'] * buffer_size)

    for i in range(task_id,len(train_set_list)):
        trSet_name = train_set_list[i]
        valSet_name = val_set_list[i]

        trSet = ngsimDataset(trSet_name)
        iterations = trSet.__len__()//args['batch_size']
        trDataloader = DataLoader(trSet,batch_size=args['batch_size'],
                                  shuffle=True,num_workers=12,collate_fn=trSet.collate_fn)

        valSet = ngsimDataset(valSet_name)
        valDataloader = DataLoader(valSet,batch_size=args['batch_size'],
                                   shuffle=True,num_workers=12,collate_fn=valSet.collate_fn)
        ## Variables holding train and validation loss values:
        # net.load_state_dict(model_init_param)
        with trange(start_epoch, trainEpochs) as t_epoch:
            for epoch_num in t_epoch:
                t_epoch.set_description("epoch %i for task %i" % (epoch_num, i))
                avg_tr_loss = 0

                for j, data in enumerate(tqdm(trDataloader)):
                    st_time = time.time()
                    hist_new, nbrs_new, fut_new, _,index_division_new = data
                    old_param = net.state_dict()
                    old_opt = optimizer.state_dict()                   
                    if args['use_cuda']:
                        hist_new = hist_new.cuda()
                        nbrs_new = nbrs_new.cuda()
                        fut_new = fut_new.cuda()

                    fut_pred_new = net(hist_new, nbrs_new, index_division_new)
                    # l = RMSE(fut_pred, fut)
                    l = maskedMSE(fut_pred_new, fut_new, torch.ones_like(fut_new).cuda())

                    optimizer.zero_grad()
                    l.backward()
                    a = torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
                    optimizer.step()
                    
                    hist_ret, nbrs_ret, fut_ret, index_division_ret = buffer.mir_retrieve(net, old_param, args['batch_size'])
                    net.load_state_dict(old_param)
                    optimizer.load_state_dict(old_opt)
                    
                    # hist_ret = []
                    if len(hist_ret):
                        if args['use_cuda']:
                            hist_ret = hist_ret.cuda()
                            nbrs_ret = nbrs_ret.cuda()
                            fut_ret = fut_ret.cuda()
                        hist = torch.cat((hist_new,hist_ret),dim=1)
                        fut = torch.cat((fut_new,fut_ret),dim=1)
                        nbrs = torch.cat((nbrs_new,nbrs_ret),dim=1)
                        index_division = copy.deepcopy(index_division_new)
                        index_start = index_division_new[-1][-1]+1
                        for index_div in index_division_ret:
                            index_division.append([index+index_start for index in index_div])
                    else:
                        hist = hist_new
                        fut = fut_new
                        nbrs = nbrs_new
                        index_division = index_division_new

                    fut_pred = net(hist, nbrs, index_division)
                    # l = RMSE(fut_pred, fut)
                    l = maskedMSE(fut_pred, fut, torch.ones_like(fut).cuda())

                    optimizer.zero_grad()
                    l.backward()
                    a = torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
                    optimizer.step()
                    avg_tr_loss += l.item()
                    train_loss.append(l.item())
                    if epoch_num==0:
                        buffer.reservior_update(hist_new,fut_new,nbrs_new,index_division_new)

                    if j%100 == 99:
                        tqdm.write('avg rmse loss:{:.4f} over {} iterations'
                                   .format(avg_tr_loss / 100, 100))
                        avg_tr_loss = 0
                plot_epoch_num = 3
                if epoch_num > plot_epoch_num:
                    plt.plot(np.arange(iterations * (epoch_num - plot_epoch_num), iterations * epoch_num),
                             train_loss[iterations * (epoch_num - plot_epoch_num):iterations * epoch_num])
                else:
                    plt.plot(np.arange((epoch_num + 1) * iterations),train_loss[:(epoch_num + 1) * iterations])
                fig_path = trial_path+'figs/task'+str(i)
                if not os.path.exists(fig_path):
                    os.makedirs(fig_path)
                plt.savefig(fig_path+'/train_loss_epoch' + str(epoch_num))
                plt.close()

                plt.plot(np.arange(len(train_loss)), train_loss)
                plt.savefig(fig_path + '/full_train_loss')
                plt.close()
                # _________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
                # sio.savemat('train_data.mat',{'real':fut.detach().cpu().numpy(),'pred':fut_pred.detach().cpu().numpy()})

                ## Validate:______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
                tqdm.write("Epoch {} complete. Calculating validation loss...".format(epoch_num+1))
                avg_val_loss = 0
                val_batch_count = 0

                for j, data in enumerate(valDataloader):
                    st_time = time.time()
                    hist, nbrs, fut, scale,index_division = data

                    if args['use_cuda']:
                        hist = hist.cuda()
                        nbrs = nbrs.cuda()
                        fut = fut.cuda()
                        scale = scale.cuda()

                    # Forward pass
                    # with torch.no_grad():
                    fut_pred = net(hist, nbrs, index_division)
                    # loss = RMSE(fut_pred*scale, fut*scale)
                    loss = maskedMSE(fut_pred*scale, fut*scale, torch.ones_like(fut).cuda())

                    avg_val_loss += loss.item()
                    val_batch_count += 1

                # sio.savemat('val_data.mat',{'real':fut.detach().cpu().numpy(),'pred':fut_pred.detach().cpu().numpy()})

                # Print validation loss and update display variables
                tqdm.write('Validation loss : {:.4f}'.format(avg_val_loss/val_batch_count))
                val_loss.append(avg_val_loss/val_batch_count)
                prev_val_loss = avg_val_loss/val_batch_count
                if prev_val_loss < best_val_loss or epoch_num == 0:
                    best_val_loss = prev_val_loss
                    best_val_model_param = net.state_dict()
                    model_path = trial_path+'trained_models/task'+str(i)
                    if not os.path.exists(model_path):
                        os.makedirs(model_path)
                    torch.save(net.state_dict(), model_path+'/bestval_model_epoch'+str(epoch_num)+'.tar')
                    tqdm.write("Best val loss updated at epoch {}!".format(epoch_num))

                # checkpoint
                if epoch_num % 3 == 0 or epoch_num == trainEpochs-1:
                    buffer.save_buffer(buffer_tmp_file+"/mir_buffer_tmp.mat")
                    checkpoint = {
                        'epoch': epoch_num,
                        'task_id': i,
                        'model': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'best_val_loss': best_val_loss,
                        'best_val_model_param': best_val_model_param
                    }
                    # torch.cuda.empty_cache()
                    torch.save(checkpoint, trial_path+'checkpoint.pkl')
                    model_tmp_path = trial_path+'trained_models_tmp/task'+str(i)
                    if not os.path.exists(model_tmp_path):
                        os.makedirs(model_tmp_path)
                    torch.save(net.state_dict(),
                               model_tmp_path+'/model_epoch' + str(epoch_num) + '.tar')
        torch.save(best_val_model_param, model_path + '/bestval_model.tar')
        # load best model param
        # reset optimizer
        # save buffer into file
        # prepare for the next task
        # net.load_state_dict(torch.load(model_path+'/bestval_model.tar'))
        net.load_state_dict(best_val_model_param)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        buffer_file = trial_path+"data/mir_buffer_"+str(i)+".mat"
        buffer.save_buffer(buffer_file)
        best_val_loss = math.inf
        best_val_model_param = None
        # evaluate on all seen task
        test_rec = torch.zeros(i+1,args['out_length'])
        for t in range(i+1):
            tsSet_name = test_set_list[t]
            tsSet = ngsimDataset(tsSet_name)
            tsDataloader = DataLoader(tsSet, batch_size=args['batch_size'],
                                      shuffle=True, num_workers=12, collate_fn=tsSet.collate_fn)

            # err_rec = []
            lossVals = torch.zeros(args['out_length']).cuda()
            counts = torch.zeros(args['out_length']).cuda()
            for data in tsDataloader:
                hist, nbrs, fut, scale, index_division = data
                if args['use_cuda']:
                    hist = hist.cuda()
                    nbrs = nbrs.cuda()
                    fut = fut.cuda()
                    scale = scale.cuda()

                # Forward pass
                with torch.no_grad():
                    fut_pred = net(hist, nbrs, index_division)
                # loss = RMSE_time(fut_pred * scale, fut * scale)
                l,c = maskedMSETest(fut_pred * scale, fut * scale,torch.ones_like(fut).cuda())
                lossVals += l.detach()
                counts += c.detach()
                # err_rec.append(loss)
            # err_rec = torch.cat(err_rec,axis=1)
            unit_scale = 0.3048
            if 'highd' in tsSet_name or 'inter5d' in tsSet_name:
                unit_scale = 1.0
            # test_rec[t, :] = err_rec.mean(1).view(1, args['out_length'])*unit_scale
            test_rec[t, :] = torch.pow(lossVals / counts,0.5)*unit_scale
            # print(torch.pow(lossVals / counts,0.5)*unit_scale)
        eval_file = trial_path+'evaluate_lifelong_till'+str(i)+'.mat'
        sio.savemat(eval_file, {'rmse': test_rec.detach().cpu().numpy()})
        tqdm.write('lifelong evaluate untill task {} result has been saved into {}'.format(i, eval_file))







