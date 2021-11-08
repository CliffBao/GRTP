from __future__ import print_function
import torch
from model import highwayNet
from utils import ngsimDataset,maskedNLL,maskedMSETest,maskedNLLTest
from torch.utils.data import DataLoader
import time
import scipy.io as sio
import numpy

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
args['train_flag'] = False

# Evaluation metric:
# metric = 'nll'  #or rmse
metric = 'rmse'

# Initialize network
net = highwayNet(args)
net.load_state_dict(torch.load('trained_models/cslstm_us1011_nei3&4_nsl_bestval_5hz_sgan201.tar'))
if args['use_cuda']:
    net = net.cuda()

tsSet = ngsimDataset('../data/TestSet-us1011-gan-smooth-nei3&4-nsl.mat')
# tsSet = ngsimDataset('data/TestSet-carla-10-nei1-maxgrid.mat')
tsDataloader = DataLoader(tsSet,batch_size=128,shuffle=True,num_workers=4,collate_fn=tsSet.collate_fn)

lossVals = torch.zeros(args['out_length']).cuda()
counts = torch.zeros(args['out_length']).cuda()


for i, data in enumerate(tsDataloader):
    st_time = time.time()
    hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask,scale,index_division = data

    # Initialize Variables
    if args['use_cuda']:
        hist = hist.cuda()
        nbrs = nbrs.cuda()
        mask = mask.cuda()
        lat_enc = lat_enc.cuda()
        lon_enc = lon_enc.cuda()
        fut = fut.cuda()
        op_mask = op_mask.cuda()
        scale = scale.cuda()
    if metric == 'nll':
        # Forward pass
        if args['use_maneuvers']:
            fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc,index_division)
            l,c = maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask)
        else:
            fut_pred = net(hist, nbrs, lat_enc, lon_enc,index_division)
            l, c = maskedNLLTest(fut_pred, 0, 0, fut, op_mask,use_maneuvers=False)
    else:
        # Forward pass
        if args['use_maneuvers']:
            fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc,index_division)
            fut_pred_max = torch.zeros_like(fut_pred[0])
            for k in range(lat_pred.shape[0]):
                lat_man = torch.argmax(lat_pred[k, :]).detach()
                lon_man = torch.argmax(lon_pred[k, :]).detach()
                indx = lon_man*3 + lat_man
                fut_pred_max[:,k,:] = fut_pred[indx][:,k,:]
            l, c = maskedMSETest(fut_pred_max, fut, op_mask)

        else:
            fut_pred = net(hist, nbrs, lat_enc, lon_enc,index_division)
            fut = fut*scale
            fut_pred[:, :, 0:2] = fut_pred[:, :, 0:2] * scale
            l, c = maskedMSETest(fut_pred, fut, op_mask)
            if i == 0:
                sio.savemat('cslstm-eval.mat',
                            {'real':fut.detach().cpu().numpy(),'pred':fut_pred.detach().cpu().numpy()})
    # print(l)
    lossVals +=l.detach()
    counts += c.detach()

if metric == 'nll':
    print(lossVals / counts)
else:
    print(torch.pow(lossVals / counts,0.5)*0.3048)   # Calculate RMSE and convert from feet to meters


