from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from utils import outputActivation
import math

class highwayNet(nn.Module):

    ## Initialization
    def __init__(self,args):
        super(highwayNet, self).__init__()

        ## Unpack arguments
        self.args = args

        ## Use gpu flag
        self.use_cuda = args['use_cuda']

        # Flag for maneuver based (True) vs uni-modal decoder (False)
        self.use_maneuvers = args['use_maneuvers']

        # Flag for train mode (True) vs test-mode (False)
        self.train_flag = args['train_flag']

        ## Sizes of network layers
        self.encoder_size = args['encoder_size']
        self.decoder_size = args['decoder_size']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.grid_size = args['grid_size']
        self.soc_conv_depth = args['soc_conv_depth']
        self.conv_3x1_depth = args['conv_3x1_depth']
        self.dyn_embedding_size = args['dyn_embedding_size']
        self.input_embedding_size = args['input_embedding_size']
        self.num_lat_classes = args['num_lat_classes']
        self.num_lon_classes = args['num_lon_classes']
        self.soc_embedding_size = (((args['grid_size'][0]-4)+1)//2)*self.conv_3x1_depth
        # self.soc_embedding_size = self.encoder_size

        ## Define network weights

        # Input embedding layer
        self.ip_emb = torch.nn.Linear(2,self.input_embedding_size)

        # Encoder LSTM
        self.enc_lstm = torch.nn.LSTM(self.input_embedding_size,self.encoder_size,1)

        # Vehicle dynamics embedding
        self.dyn_emb = torch.nn.Linear(self.encoder_size,self.dyn_embedding_size)
        self.nbr_emb = torch.nn.Linear(2*self.in_length,self.encoder_size)

        # Convolutional social pooling layer and social embedding layer
        self.soc_conv = torch.nn.Conv2d(self.encoder_size,self.soc_conv_depth,3)
        self.conv_3x1 = torch.nn.Conv2d(self.soc_conv_depth, self.conv_3x1_depth, (3,1))
        self.soc_maxpool = torch.nn.MaxPool2d((2,1),padding = (1,0))

        # FC social pooling layer (for comparison):
        # self.soc_fc = torch.nn.Linear(self.soc_conv_depth * self.grid_size[0] * self.grid_size[1], (((args['grid_size'][0]-4)+1)//2)*self.conv_3x1_depth)

        # Decoder LSTM
        if self.use_maneuvers:
            self.dec_lstm = torch.nn.LSTM(self.soc_embedding_size + self.dyn_embedding_size + self.num_lat_classes + self.num_lon_classes, self.decoder_size)
            self.out_linear = torch.nn.Linear(self.dyn_embedding_size + self.dyn_embedding_size + self.num_lat_classes + self.num_lon_classes, self.decoder_size)
        else:
            self.dec_lstm = torch.nn.LSTM(self.dyn_embedding_size + self.dyn_embedding_size, self.decoder_size)
            self.out_linear = torch.nn.Linear(self.dyn_embedding_size + self.dyn_embedding_size, self.decoder_size)

        # Output layers:
        # self.op = torch.nn.Linear(self.decoder_size,2*self.out_length)
        self.op = torch.nn.Linear(self.decoder_size,2)
        self.op_lat = torch.nn.Linear(self.soc_embedding_size + self.dyn_embedding_size, self.num_lat_classes)
        self.op_lon = torch.nn.Linear(self.soc_embedding_size + self.dyn_embedding_size, self.num_lon_classes)

        # Activations:
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def make_index(self,index_list):
        len_list = [[len(i)] * len(i) for i in index_list]
        index_split = sum(len_list, [])
        index_1122 = np.repeat(sum(index_list, []), index_split)

        len_list = [len(i) for i in index_list]
        index_1212 = sum([i for i in index_list for k in range(len(i))], [])

        # len_list = sum([[len(i)]*len(i) for i in index_list],[])
        # index_repeated = [None] * len(len_list)
        len_list = [len(i) ** 2 for i in index_list]
        index_repeated = [None] * len(len_list)
        index_repeated[0] = np.linspace(0, len_list[0] - 1, len_list[0]).astype(int)
        count = len_list[0]
        for i, index in enumerate(len_list):
            if i == len(len_list) - 1:
                break
            index_repeated[i + 1] = np.linspace(count, count + len_list[i + 1] - 1, len_list[i + 1]).astype(int)
            count += len_list[i + 1]
        return index_1212, index_1122, index_repeated

    def forward(self, hist, nbrs, lat_enc, lon_enc, index_division):
        hero_index = np.arange(hist.shape[1]).tolist()
        index_len = [len(i) for i in index_division]
        hero_repeated = np.repeat(hero_index, index_len)

        _, (hist_enc, _) = self.enc_lstm(self.leaky_relu(self.ip_emb(hist)))
        # _, (nbrs_enc, _) = self.enc_lstm(self.leaky_relu(self.ip_emb(nbrs)))
        # nbrs_enc = nbrs_enc.squeeze(0)
        hist_enc = hist_enc.squeeze(0)

        relative = hist[:,hero_repeated, :] - nbrs
        rela_enc = self.leaky_relu(self.nbr_emb(relative.permute(1,0,2).contiguous().view(-1,2*self.in_length)))
        hist_enc = self.leaky_relu(self.dyn_emb(hist_enc))

        ## Forward pass nbrs
        # index_1212,index_1122,index_repeated = self.make_index(index_division)
        # if not self.train_flag:
        #     print(index_division)
        scene_pooled = torch.cat([rela_enc[index, :].max(0)[0].unsqueeze(0) for index in index_division], dim=0)
        scene_pooled = self.leaky_relu(self.dyn_emb(scene_pooled))
        ## Masked scatter
        # print(hist_enc.shape,scene_pooled.shape)
        enc = torch.cat((hist_enc,scene_pooled),1)
        # enc = hist_enc

        if self.use_maneuvers:
            ## Maneuver recognition:
            lat_pred = self.softmax(self.op_lat(enc))
            lon_pred = self.softmax(self.op_lon(enc))

            if self.train_flag:
                ## Concatenate maneuver encoding of the true maneuver
                enc = torch.cat((enc, lat_enc, lon_enc), 1)
                fut_pred = self.decode(enc)
                return fut_pred, lat_pred, lon_pred
            else:
                fut_pred = []
                ## Predict trajectory distributions for each maneuver class
                for k in range(self.num_lon_classes):
                    for l in range(self.num_lat_classes):
                        lat_enc_tmp = torch.zeros_like(lat_enc)
                        lon_enc_tmp = torch.zeros_like(lon_enc)
                        lat_enc_tmp[:, l] = 1
                        lon_enc_tmp[:, k] = 1
                        enc_tmp = torch.cat((enc, lat_enc_tmp, lon_enc_tmp), 1)
                        fut_pred.append(self.decode(enc_tmp))
                return fut_pred, lat_pred, lon_pred
        else:
            fut_pred = self.decode(enc)
        return fut_pred

    def decode(self,enc):
        # print('enc size', enc.shape)
        enc = enc.unsqueeze(0).repeat(self.out_length, 1, 1)
        # print('after repeat ', enc.shape)
        h_dec, _ = self.dec_lstm(enc)
        # print('after dec', h_dec.shape)
        h_dec = h_dec.permute(1, 0, 2)
        fut_pred = self.op(h_dec)
        fut_pred = fut_pred.permute(1, 0, 2)
        fut_pred = outputActivation(fut_pred)
        return fut_pred
    # def decode(self,enc):
    #     h_dec = self.out_linear(enc)
    #     # print('after dec', h_dec.shape)
    #     fut_pred = self.op(h_dec)
    #     fut_pred = fut_pred.view(-1,2,self.out_length).permute(2, 0, 1)
    #     fut_pred = outputActivation(fut_pred)
    #     return fut_pred





