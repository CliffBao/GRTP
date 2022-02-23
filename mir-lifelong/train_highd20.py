from __future__ import print_function
import torch
from model import highwayNet
from utils import ngsimDataset,maskedMSE,maskedMSETest
from torch.utils.data import DataLoader
import scipy.io as sio
import time
import math


## Network Arguments
args = {}
args['use_cuda'] = True
args['encoder_size'] = 64
args['decoder_size'] = 128
args['in_length'] = 16
args['out_length'] = 25
args['dyn_embedding_size'] = 32
args['input_embedding_size'] = 32
args['grid_size'] = (13,3)
args['batch_size'] = 128

# Initialize network
net = highwayNet(args)
# net.load_state_dict(torch.load('trained_models/cslstm_m.tar'))
if args['use_cuda']:
    net = net.cuda()

## Initialize optimizer
trainEpochs = 8
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
batch_size = 128
crossEnt = torch.nn.BCELoss()

def RMSE(real,pred):
    err = torch.sqrt(torch.sum(torch.pow(real-pred,2),2))
    err = torch.mean(err,axis=1)
    return torch.mean(err,axis=0)


def RMSE_time(real,pred):
    err = torch.sqrt(torch.sum(torch.pow(real-pred,2),2))
    err = torch.mean(err,dim=1,keepdim=True)
    return err

## Initialize data loaders
trSet = ngsimDataset('data/TrainSet-highd20-gan-pos-nsl.mat')
valSet = ngsimDataset('data/ValSet-highd20-gan-pos-nsl.mat')
trDataloader = DataLoader(trSet,batch_size=batch_size,shuffle=True,num_workers=4,collate_fn=trSet.collate_fn)
valDataloader = DataLoader(valSet,batch_size=batch_size,shuffle=True,num_workers=4,collate_fn=valSet.collate_fn)


## Variables holding train and validation loss values:
train_loss = []
best_val_loss = 99999.9
val_loss = []
prev_val_loss = math.inf

for epoch_num in range(trainEpochs):

    # Variables to track training performance:
    avg_tr_loss = 0
    avg_tr_time = 0

    for i, data in enumerate(trDataloader):

        st_time = time.time()
        hist, nbrs, fut, scale, index_div = data
        # sio.savemat('nbrs.mat',{'hist':nbrs.permute(1,2,0).contiguous().view(-1,32).numpy()})
        if args['use_cuda']:
            hist = hist.cuda()
            nbrs = nbrs.cuda()
            fut = fut.cuda()
            scale = scale.cuda()

        # Forward pass
        fut_pred = net(hist, nbrs, index_div)
        # fut = fut*scale
        # fut_pred[:,:,0:2] = fut_pred[:,:,0:2]*scale
        l = maskedMSE(fut_pred, fut, torch.ones_like(fut))
        # l = RMSE(fut_pred, fut)

        # Backprop and update weights
        optimizer.zero_grad()
        l.backward()
        a = torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        optimizer.step()

        # Track average train loss and average train time:
        batch_time = time.time()-st_time
        avg_tr_loss += l.item()
        avg_tr_time += batch_time

        if i%100 == 99:
            eta = avg_tr_time/100*(len(trSet)/batch_size-i)
            print("Epoch no:",epoch_num+1,"| Epoch progress(%):",format(i/(len(trSet)/batch_size)*100,'0.2f'), "| Avg train loss:",format(avg_tr_loss/100,'0.4f'), "| Validation loss prev epoch",format(prev_val_loss,'0.4f'), "| ETA(s):",int(eta))
            train_loss.append(avg_tr_loss/100)
            avg_tr_loss = 0
            avg_lat_acc = 0
            avg_lon_acc = 0
            avg_tr_time = 0
    # _________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________



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
        hist, nbrs, fut, scale, index_div = data
        if args['use_cuda']:
            hist = hist.cuda()
            nbrs = nbrs.cuda()
            fut = fut.cuda()
            scale = scale.cuda()

        # Forward pass
        fut_pred = net(hist, nbrs, index_div)
        l = maskedMSE(fut_pred*scale, fut*scale, torch.ones_like(fut))
        # l = RMSE(fut_pred, fut)

        avg_val_loss += l.item()
        val_batch_count += 1

    print(avg_val_loss/val_batch_count)

    # Print validation loss and update display variables
    print('Validation loss :',format(avg_val_loss/val_batch_count,'0.4f'),"| Val Acc:",format(avg_val_lat_acc/val_batch_count*100,'0.4f'),format(avg_val_lon_acc/val_batch_count*100,'0.4f'))
    val_loss.append(avg_val_loss/val_batch_count)
    prev_val_loss = avg_val_loss/val_batch_count
    if avg_val_loss < best_val_loss or epoch_num == 0:
        best_val_loss = avg_val_loss
        torch.save(net.state_dict(), 'trained_models/cslstm_highd20p_nsl_5hz_bestval.tar')
        print("Best val loss updated! current best val loss is at epoch: ", epoch_num + 1)



