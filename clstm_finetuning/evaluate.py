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
args['encoder_size'] = 64
args['decoder_size'] = 128
args['in_length'] = 16
args['out_length'] = 25
args['dyn_embedding_size'] = 32
args['input_embedding_size'] = 32

# Evaluation metric:
# metric = 'nll'  #or rmse
metric = 'rmse'

# Initialize network
net = highwayNet(args)
test_model = 'trained_models/cslstm_us1011_i801_highd20p_inter5dclu_us1012_bestval.tar'
net.load_state_dict(torch.load(test_model))
if args['use_cuda']:
    net = net.cuda()

test_data = 'data/TestSet-us1012-gan-smooth-nei3&4-nsl.mat'
# test_data = 'data/TestSet-i801-gan-smooth-nei3&4-nsl.mat'
# test_data = 'data/TestSet-highd20-gan-pos-nsl.mat'
# test_data = 'data/TestSet-inter5d-gan-lane-clustered.mat'
tsSet = ngsimDataset(test_data)
tsDataloader = DataLoader(tsSet,batch_size=128,shuffle=True,num_workers=4,collate_fn=tsSet.collate_fn)

lossVals = torch.zeros(args['out_length']).cuda()
counts = torch.zeros(args['out_length']).cuda()

print('test model: ',test_model,';test_data:',test_data)
for i, data in enumerate(tsDataloader):
    st_time = time.time()
    hist, nbrs, fut, op_mask,scale,index_div = data

    # Initialize Variables
    if args['use_cuda']:
        hist = hist.cuda()
        nbrs = nbrs.cuda()
        fut = fut.cuda()
        op_mask = op_mask.cuda()
        scale = scale.cuda()

    fut_pred = net(hist, nbrs, index_div)
    fut = fut*scale
    fut_pred[:, :, 0:2] = fut_pred[:, :, 0:2] * scale
    l, c = maskedMSETest(fut_pred, fut, op_mask)
    if i == 0:
        pred = torch.reshape(fut_pred, (args['out_length'], fut_pred.shape[1]*fut_pred.shape[2]))
        real = torch.reshape(fut, (args['out_length'], fut.shape[1]*fut.shape[2]))
        pred = pred.cpu().detach().numpy()
        real = real.cpu().detach().numpy()

        sio.savemat('cslstm-eval.mat',
                    {'predict': pred, 'fut': real})
    # print(l)
    lossVals +=l.detach()
    counts += c.detach()

unit_converter = 0.3048
if 'highd' in test_data or 'inter' in test_data:
    unit_converter = 1.0
if metric == 'nll':
    print(lossVals / counts)
else:
    print(torch.pow(lossVals / counts,0.5)*unit_converter)   # Calculate RMSE and convert from feet to meters


