from __future__ import print_function
import torch
from model import highwayNet
from utils import ngsimDataset,maskedMSETest
from torch.utils.data import DataLoader
import time
import scipy.io as sio
import numpy

## Network Arguments
args = {}
args['use_cuda'] = True
args['encoder_size'] = 64
args['decoder_size'] = 128
args['in_length'] = 16
args['out_length'] = 25
args['dyn_embedding_size'] = 32
args['grid_size'] = (13,3)
args['batch_size'] = 128
args['input_embedding_size'] = 32

# Evaluation metric:
# metric = 'nll'  #or rmse
metric = 'rmse'

# Initialize network
net = highwayNet(args)
# net.load_state_dict(torch.load('trained_models/cslstm_highd20p_nsl_5hz_bestval.tar'))
net.load_state_dict(torch.load('trained_models/task0/bestval_model.tar'))

if args['use_cuda']:
    net = net.cuda()

tsSet_name = 'data/TestSet-highd20-gan-pos-nsl.mat'
tsSet = ngsimDataset(tsSet_name)
tsDataloader = DataLoader(tsSet,batch_size=128,shuffle=True,num_workers=4,collate_fn=tsSet.collate_fn)

lossVals = torch.zeros(args['out_length']).cuda()
counts = torch.zeros(args['out_length']).cuda()

def RMSE_time(real,pred):
    err = torch.sqrt(torch.sum(torch.pow(real-pred,2),2))
    err = torch.mean(err,dim=1,keepdim=True)
    return err

err_rec = []
for i, data in enumerate(tsDataloader):
    st_time = time.time()
    hist, nbrs, fut, scale,index_div = data

    # Initialize Variables
    if args['use_cuda']:
        hist = hist.cuda()
        nbrs = nbrs.cuda()
        fut = fut.cuda()
        scale = scale.cuda()

    fut_pred = net(hist, nbrs, index_div)
    fut = fut*scale
    fut_pred[:, :, 0:2] = fut_pred[:, :, 0:2] * scale
    l, c = maskedMSETest(fut_pred, fut, torch.ones_like(fut).cuda())
    # l = RMSE_time(fut_pred, fut)
    if i == 0:
        pred = torch.reshape(fut_pred, (args['out_length'], fut_pred.shape[1]*fut_pred.shape[2]))
        real = torch.reshape(fut, (args['out_length'], fut.shape[1]*fut.shape[2]))
        pred = pred.cpu().detach().numpy()
        real = real.cpu().detach().numpy()

        sio.savemat('cslstm-eval.mat',
                    {'predict': pred, 'fut': real})
    # print(l)
    lossVals += l.detach()
    counts += c.detach()
# err_rec = torch.cat(err_rec, axis=1)
unit_scale = 0.3048
if 'highd' in tsSet_name or 'inter5d' in tsSet_name:
    unit_scale = 1.0
# err = err_rec.mean(1).view(1, args['out_length'])*unit_scale
# print(err)
print(torch.pow(lossVals / counts,0.5)*unit_scale)

