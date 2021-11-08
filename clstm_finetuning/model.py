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

        ## Sizes of network layers
        self.encoder_size = args['encoder_size']
        self.decoder_size = args['decoder_size']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.dyn_embedding_size = args['dyn_embedding_size']
        self.input_embedding_size = args['input_embedding_size']

        ## Define network weights
        # Input embedding layer
        self.ip_emb = torch.nn.Linear(2,self.input_embedding_size)

        # Encoder LSTM
        self.enc_lstm = torch.nn.LSTM(self.input_embedding_size,self.encoder_size,1)

        # Vehicle dynamics embedding
        self.dyn_emb = torch.nn.Linear(self.encoder_size,self.dyn_embedding_size)
        self.nbr_emb = torch.nn.Linear(2*self.in_length,self.encoder_size)

        # Decoder LSTM
        self.dec_lstm = torch.nn.LSTM(self.dyn_embedding_size + self.dyn_embedding_size, self.decoder_size)
        self.out_linear = torch.nn.Linear(self.dyn_embedding_size + self.dyn_embedding_size, self.decoder_size)

        # Output layers:
        # self.op = torch.nn.Linear(self.decoder_size,2*self.out_length)
        self.op = torch.nn.Linear(self.decoder_size,2)

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

    def forward(self, hist, nbrs, index_division):
        hero_index = np.arange(hist.shape[1]).tolist()
        index_len = [len(i) for i in index_division]
        hero_repeated = np.repeat(hero_index, index_len)

        _, (hist_enc, _) = self.enc_lstm(self.leaky_relu(self.ip_emb(hist)))
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






