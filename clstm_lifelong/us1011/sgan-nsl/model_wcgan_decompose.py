from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import math
from jit_gru import JitGRU
from jit_gru_ln import JitGRULN
# from indrnn import IndRNNv2
import time
from scipy.spatial.distance import pdist, squareform
from torch.distributions.multivariate_normal import MultivariateNormal
import scipy.io as sio


class highwayNet_d(nn.Module):
    ## Initialization
    def __init__(self, args):
        super(highwayNet_d, self).__init__()

        ## Unpack arguments
        self.args = args
        ## Use gpu flag
        self.use_cuda = args['use_cuda']
        self.encoder_size = args['encoder_size']
        ## Sizes of network layers
        self.in_length = args['in_length']
        self.input_embedding_size = args['input_embedding_size_d']
        self.out_length = args['out_length']
        self.batch_size = args['batch_size']
        self.class_num = args['class_num']
        ## Define network weights
        self.recurrent_max = pow(2,1.0/self.in_length)
        self.recurrent_min = pow(1/2,1.0/self.in_length)
        self.n_layers = 1

        self.recurrent_inits = []
        for _ in range(self.n_layers - 1):
            self.recurrent_inits.append(
                lambda w: nn.init.uniform_(w, 0, 1)
            )
        self.recurrent_inits.append(lambda w: nn.init.uniform_(
            w, 1-1e-10, 1+1e-10))
        # self.traj_lstm = IndRNNv2(
        #     2*self.encoder_size, self.encoder_size, self.n_layers, batch_norm=False,
        #     hidden_max_abs=self.recurrent_max, hidden_min_abs=-self.recurrent_max, batch_first=False,
        #     bidirectional=False, recurrent_inits=self.recurrent_inits,
        #     gradient_clip=5
        # )
        # self.traj_lstm_v = IndRNNv2(
        #     2*self.encoder_size, self.encoder_size, self.n_layers, batch_norm=False,
        #     hidden_max_abs=self.recurrent_max, hidden_min_abs=-self.recurrent_max, batch_first=False,
        #     bidirectional=False, recurrent_inits=self.recurrent_inits,
        #     gradient_clip=5
        # )

        self.ip_emb_d = torch.nn.Linear(2, self.input_embedding_size)
        # self.ip_emb_v = torch.nn.Linear(2, self.input_embedding_size)
        # self.nbr_emb_d = torch.nn.Linear(2*self.traj_num, self.input_embedding_size)
        # self.ip_emb_d = torch.nn.Linear(2*self.traj_num, self.input_embedding_size)
        # self.spatial_emb = torch.nn.Linear(2,self.input_embedding_size)
        # self.global_emb = torch.nn.Sequential(
        #     torch.nn.Linear(4, self.input_embedding_size),
        #     torch.nn.SELU(inplace=True),
        #     torch.nn.Linear(self.input_embedding_size,self.encoder_size)
        # )
        # Input embedding layer
        # self.cond_emb = torch.nn.Sequential(
        #     torch.nn.Linear(2, self.input_embedding_size),
        #     torch.nn.LeakyReLU(0.1, inplace=True),
        #     torch.nn.Linear(self.input_embedding_size, self.encoder_size),
        #     torch.nn.LeakyReLU(0.1, inplace=True)
        # )
        # self.cond_emb_back = torch.nn.Sequential(
        #     torch.nn.Linear(2, self.input_embedding_size),
        #     torch.nn.LeakyReLU(0.1, inplace=True),
        #     torch.nn.Linear(self.input_embedding_size, self.encoder_size),
        #     torch.nn.LeakyReLU(0.1, inplace=True)
        # )
        # self.cond_emb_full = torch.nn.Sequential(
        #     torch.nn.Linear(2, self.input_embedding_size),
        #     torch.nn.LeakyReLU(0.1, inplace=True),
        #     torch.nn.Linear(self.input_embedding_size, self.encoder_size),
        #     torch.nn.LeakyReLU(0.1, inplace=True)
        # )

        # self.cond_emb_traj_v = torch.nn.Sequential(
        #     torch.nn.Linear(2*self.traj_num, self.input_embedding_size),
        #     torch.nn.LeakyReLU(0.1, inplace=True),
        #     torch.nn.Linear(self.input_embedding_size, self.encoder_size),
        #     torch.nn.LeakyReLU(0.1, inplace=True)
        # )
        # self.enc_lstm = IndRNNv2(
        #     self.input_embedding_size, self.encoder_size, self.n_layers, batch_norm=False,
        #     hidden_max_abs=self.recurrent_max, hidden_min_abs=-self.recurrent_max, batch_first=False,
        #     bidirectional=False, recurrent_inits=self.recurrent_inits,
        #     gradient_clip=5
        # )
        # self.enc_lstm_back = IndRNNv2(
        #     self.input_embedding_size, self.encoder_size, self.n_layers, batch_norm=False,
        #     hidden_max_abs=self.recurrent_max, hidden_min_abs=-self.recurrent_max, batch_first=False,
        #     bidirectional=False, recurrent_inits=self.recurrent_inits,
        #     gradient_clip=5
        # )
        # self.traj_lstm = JitGRU(2*self.encoder_size,self.encoder_size,1)
        self.enc_lstm_back = JitGRU(self.input_embedding_size,self.encoder_size,1)
        self.enc_lstm = JitGRU(self.input_embedding_size,self.encoder_size,1)
        # self.enc_lstm_v = IndRNNv2(
        #     self.input_embedding_size, self.encoder_size, self.n_layers, batch_norm=False,
        #     hidden_max_abs=self.recurrent_max, hidden_min_abs=-self.recurrent_max, batch_first=False,
        #     bidirectional=False, recurrent_inits=self.recurrent_inits,
        #     gradient_clip=5
        # )
        # self.enc_lstm_v_back = IndRNNv2(
        #     self.input_embedding_size, self.encoder_size, self.n_layers, batch_norm=False,
        #     hidden_max_abs=self.recurrent_max, hidden_min_abs=-self.recurrent_max, batch_first=False,
        #     bidirectional=False, recurrent_inits=self.recurrent_inits,
        #     gradient_clip=5
        # )
        # self.traj_lstm_v = JitGRU(2*self.encoder_size,self.encoder_size,1)
        # self.spatial_lstm = JitGRU(self.input_embedding_size,self.encoder_size,1)
        # self.op_d = torch.nn.Linear(self.encoder_size*2,1)
        # self.cat_emb = torch.nn.Linear(self.encoder_size, self.encoder_size//2)
        self.op_d = torch.nn.Sequential(
            torch.nn.Linear(self.encoder_size, self.encoder_size // 4),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Linear(self.encoder_size // 4, self.class_num),
        )

        self.spatial_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * self.in_length, self.encoder_size),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Linear(self.encoder_size, self.encoder_size//2),
            torch.nn.LeakyReLU(0.1, inplace=True),
        )
        # self.spatial_mlp_v = torch.nn.Sequential(
        #     torch.nn.Linear(2 * self.in_length, self.encoder_size * 2),
        #     torch.nn.LeakyReLU(0.1, inplace=True),
        #     torch.nn.Linear(self.encoder_size * 2, self.encoder_size),
        #     torch.nn.LeakyReLU(0.1, inplace=True)
        # )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.encoder_size, self.encoder_size//2),
            torch.nn.LeakyReLU(0.1, inplace=True)
        )
        # self.hero_mlp = torch.nn.Sequential(
        #     torch.nn.Linear(self.encoder_size * 2, self.encoder_size * 4),
        #     torch.nn.LeakyReLU(0.1, inplace=True),
        #     torch.nn.Linear(self.encoder_size * 4, self.encoder_size),
        #     torch.nn.LeakyReLU(0.1, inplace=True)
        # )
        # self.hero_mlp_v = torch.nn.Sequential(
        #     torch.nn.Linear(self.encoder_size * 2, self.encoder_size * 4),
        #     torch.nn.LeakyReLU(0.1, inplace=True),
        #     torch.nn.Linear(self.encoder_size * 4, self.encoder_size),
        #     torch.nn.LeakyReLU(0.1, inplace=True)
        # )
        # self.nbr_mlp = torch.nn.Sequential(
        #     torch.nn.Linear(2 * self.encoder_size, self.encoder_size * 4),
        #     torch.nn.LeakyReLU(0.1, inplace=True),
        #     torch.nn.Linear(self.encoder_size * 4, self.encoder_size),
        #     torch.nn.LeakyReLU(0.1, inplace=True)
        # )
        # self.nbr_mlp_v = torch.nn.Sequential(
        #     torch.nn.Linear(2 * self.encoder_size, self.encoder_size * 4),
        #     torch.nn.LeakyReLU(0.1, inplace=True),
        #     torch.nn.Linear(self.encoder_size * 4, self.encoder_size),
        #     torch.nn.LeakyReLU(0.1, inplace=True)
        # )

        # self.time_op = torch.nn.Linear(self.out_length, 1)
        # self.conv_emb = torch.nn.Conv1d(2,self.input_embedding_size,kernel_size=4)
        # self.op_d = torch.nn.Linear(self.encoder_size,1)

        # Activations:
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.relu = torch.nn.ReLU()
        self.selu = torch.nn.SELU(inplace=True)
        self.softmax = torch.nn.Softmax(dim=1)
        self.sigmoid = torch.nn.Sigmoid()

    def diff(self, data, order):
        for i in range(order):
            data = data[1:, :, :] - data[:-1, :, :]
        data_imple = data[0, :, :].unsqueeze(0).repeat(order, 1, 1)
        data = torch.cat((data_imple, data), 0)
        return data

    def make_index2(self, index_list):
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

        return torch.LongTensor(index_1212).cuda(), torch.LongTensor(index_1122).cuda(), index_repeated

    ## Forward Pass
    def forward(self, scene, condition, hero_pos_index, nbrs_pos_index,index_div):
        # version 11.19
        # assert hero_pos_index.size(0) == self.batch_size
        # assert hero_pos_index.size(0) * (self.traj_num - 1) == nbrs_pos_index.size(0)

        scene = scene.view(-1, 2, self.in_length).permute(2, 0, 1)
        # condition = condition.view(-1, 2)
        # condition_enc = self.cond_emb(condition)
        # condition_enc_bk = self.cond_emb_back(condition)
        # condition_enc_full = self.cond_emb_full(condition)

        index_1122, index_1212, index_repeated = self.make_index2(index_div)
        scene_1122 = scene[:, index_1122, :]
        scene_1212 = scene[:, index_1212, :]
        scene_relative = scene_1122 - scene_1212
        scene_relative = scene_relative.permute(1, 2, 0).contiguous().view(-1, 2 * self.in_length)
        # 对于浮点数而言，可能存在相同的数相减之后加起来也不为0，精度问题。
        relative_index = torch.nonzero(index_1122-index_1212, as_tuple=False)
        relative_index = relative_index.squeeze(1)
        scene_relative = scene_relative[relative_index, :]
        # 复制之后，又去除了自身与自身的作差，因此序号也要改变
        # print(index_repeated)
        for id, idx in enumerate(index_repeated):
            if id == 0:
                idx_min = 0
            else:
                idx_min = index_repeated[id-1][-1]+1
            idx_max = idx_min + len(idx) - math.sqrt(len(idx))
            idx = np.arange(idx_min, idx_max).astype(int)
            index_repeated[id] = idx.tolist()
        # index_repeated = torch.LongTensor(index_repeated).cuda()
        # print(index_repeated)
        # scene_relative = scene_relative.view(scene.shape[1], self.traj_num - 1, 2 * self.in_length)

        scene_emb = self.leaky_relu(self.ip_emb_d(scene))
        _, scene_enc = self.enc_lstm(scene_emb)
        scene_enc = scene_enc.squeeze(0)
        scene_emb_bk = torch.flip(scene_emb,dims=[0])
        _, scene_enc_bk = self.enc_lstm_back(scene_emb_bk)
        scene_enc_bk = scene_enc_bk.squeeze(0)
        scene_enc_ave = (scene_enc+scene_enc_bk)/2
        sequential_enc = self.mlp(scene_enc_ave)

        relative_enc = self.spatial_mlp(scene_relative)
        # relative_enc = torch.mean(relative_enc, dim=1)
        # for id, idx in enumerate(index_div):
        #     idx = np.array(idx) - id
        #     idx = np.arange(idx[0], idx[-1])
        #     index_div[id] = idx.tolist()
        relative_enc_pooled = torch.cat([relative_enc[index, :].
                                        view(len(index_div[id]),len(index_div[id])-1,-1).mean(1)
                                         for id,index in enumerate(index_repeated)], dim=0)
        full_enc = torch.cat((sequential_enc, relative_enc_pooled), dim=1)
        x_logit = self.op_d(full_enc)
        # sio.savemat('encoder.mat',{'scene_enc':scene_enc.detach().cpu().numpy(),
        #                            'scene_enc_bk':scene_enc_bk.detach().cpu().numpy(),
        #                            'sequential_enc':sequential_enc.detach().cpu().numpy(),
        #                            'relative_enc':relative_enc.detach().cpu().numpy(),
        #                            'condition':condition.detach().cpu().numpy(),
        #                            'x_logit':x_logit.detach().cpu().numpy(),
        #                            'scene':scene.permute(1,0,2).contiguous().view(-1,2*self.in_length).detach().cpu().numpy()})
        return x_logit,full_enc


class highwayNet_g_compose(nn.Module):
    ## Initialization
    def __init__(self, args):
        ## Unpack arguments
        super(highwayNet_g_compose, self).__init__()
        self.args = args
        ## Use gpu flag
        self.use_cuda = args['use_cuda']
        # Flag for maneuver based (True) vs uni-modal decoder (False)
        self.input_embedding_size_g = args['input_embedding_size_g']
        # Flag for train mode (True) vs test-mode (False)

        ## Sizes of network layers

        self.out_length = args['out_length']
        self.batch_size = args['batch_size']
        self.latent_dim = args['latent_dim']
        self.encoder_size = args['encoder_size']
        self.wiener_mu,self.wiener_cov = self.wiener_process_sampling_param()

        self.x = np.linspace(0., 3.0, num=self.out_length).reshape(-1, 1)
        self.gp_mu = np.zeros(self.x.shape[0])
        self.gp_cov = self.kernel(self.x,self.x)
        # enc lstm or direct sampling noise
        # self.ip_emb_g = torch.nn.Linear(2, self.input_embedding_size_g)
        # linear
        def block(in_feat, out_feat):
            layers = [torch.nn.Linear(in_feat, out_feat), torch.nn.LayerNorm(out_feat, elementwise_affine=False),
                      torch.nn.LeakyReLU(0.1)]
            return layers

        self.ip_emb_g = torch.nn.Sequential(
            *block(2*self.latent_dim,self.input_embedding_size_g//2),
            *block(self.input_embedding_size_g//2,self.input_embedding_size_g),
            # *block(self.channel_size,self.channel_size)
        )

        # self.cond_emb = torch.nn.Sequential(
        #     *block(2 * self.traj_num - 2, self.latent_dim),
        #     *block(self.latent_dim, self.latent_dim*2)
        # )
        self.cond_emb = torch.nn.Sequential(
            *block(2, self.input_embedding_size_g),
            *block(self.input_embedding_size_g, self.encoder_size)
        )
        self.cond_emb_back = torch.nn.Sequential(
            *block(2, self.input_embedding_size_g),
            *block(self.input_embedding_size_g, self.encoder_size)
        )
        # self.enc_lstm = torch.nn.GRU(self.input_embedding_size_g,self.encoder_size,bidirectional=True)
        self.enc_lstm = JitGRULN(self.input_embedding_size_g, self.encoder_size, 1)
        self.enc_lstm_back = JitGRULN(self.input_embedding_size_g, self.encoder_size, 1)
        # self.enc_lstm_spatial = torch.nn.GRU(self.input_embedding_size_g, self.encoder_size)
        # self.op_spatial = torch.nn.Sequential(
        #     torch.nn.Linear(2*self.traj_num-2, self.input_embedding_size_g),
        #     torch.nn.LeakyReLU(0.1, inplace=True),
        #     torch.nn.Linear(self.input_embedding_size, self.encoder_size),
        #     torch.nn.LeakyReLU(0.1, inplace=True)
        # )
        self.mlp = nn.Sequential(
            *block(self.latent_dim, self.latent_dim),
            # *block(2*self.latent_dim, self.encoder_size),
            # *block(self.encoder_size,self.input_embedding_size_g//2),
        )
        self.sequential_mlp = nn.Sequential(
            *block(self.encoder_size,self.latent_dim),
        )
        self.op_g = torch.nn.Sequential(
            *block(2*self.latent_dim, self.latent_dim//2),
            torch.nn.Linear(self.latent_dim//2, 2),
        )

        # Activations:
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.relu = torch.nn.ReLU()
        self.selu = torch.nn.SELU(inplace=True)
        self.softmax = torch.nn.Softmax(dim=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

        # self.model = torch.nn.Sequential(
        #     torch.nn.Conv2d(self.encoder_size, self.encoder_size//2,3),
        #     torch.nn.LeakyReLU(0.1),
        #     torch.nn.ConvTranspose2d(self.encoder_size//2,self.encoder_size,3),
        #     torch.nn.LeakyReLU(0.1)
        # )
    def diff(self, data, order):
        for i in range(order):
            data = data[1:, :, :] - data[:-1, :, :]
        data_imple = data[0, :, :].unsqueeze(0).repeat(order, 1, 1)
        data = torch.cat((data_imple, data), 0)
        return data

    def wiener_process_sampling_param(self):
        # wiener_mu = np.zeros(2*self.traj_num)
        xy_cov = 0.8*np.ones((2,2))+0.2*np.eye(2)
        # traj_cov = 0.2*np.ones((self.traj_num,self.traj_num))+0.8*np.eye(self.traj_num)
        # xy_cov_cmp = np.tile(xy_cov,(self.traj_num,self.traj_num))
        # traj_cov_cmp = traj_cov.repeat(2,axis=0).repeat(2,axis=1)
        # wiener_cov = xy_cov_cmp*traj_cov_cmp
        # X = np.linspace(0., self.latent_dim//8, num=self.latent_dim).reshape(-1, 1)
        #
        # # Mean and covariance of the prior
        wiener_mu = np.zeros(2)
        # wiener_cov = self.kernel(X, X)
        wiener_cov = xy_cov
        return wiener_mu,wiener_cov

    def sample_noise(self,x,y):
        sample = np.float32(np.random.normal(size=[x,y]))
        sample = torch.from_numpy(sample)
        if self.use_cuda:
            sample = sample.cuda()
        return sample

    def kernel(self,X1, X2, l=1.0, sigma_f=1.0):
        """
        Isotropic squared exponential kernel.

        Args:
            X1: Array of m points (m x d).
            X2: Array of n points (n x d).

        Returns:
            (m x n) matrix.
        """
        sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)
        return sigma_f ** 2 * np.exp(-0.5 / l ** 2 * sqdist)

    def wiener_sample(self):
        # mu = np.zeros(2)
        #         # cov = np.array([[1, 0.6], [0.6, 1]])
        #         # dW = []
        #         # for i in range(self.traj_num):
        #         #     dW_i = 1 / np.sqrt(self.out_length) * np.random.multivariate_normal(mu, cov, (self.batch_size, self.out_length))
        #         #     dW_i = torch.from_numpy(np.float32(dW_i))
        #         #     if self.use_cuda:
        #         #         dW_i = dW_i.cuda()
        #         #     dW.append(dW_i)
        #         # dW = torch.cat(dW,dim=2)
        # mu,cov = self.wiener_process_sampling_param()
        dW = 1 / np.sqrt(self.out_length) * \
             np.random.multivariate_normal(self.wiener_mu,
                                           self.wiener_cov,
                                           (self.batch_size, self.latent_dim*2))
        dW[:, 0, :, :] = self.wiener_mu
        dW = torch.from_numpy(np.float32(dW)).cuda()

        sample = torch.cumsum(dW, dim=1)
        sample = torch.cumsum(sample, dim=1)
        sample = torch.cumsum(sample, dim=1)
        # sample = torch.from_numpy(sample)

        return sample

    def gp_sample(self, sample_size):
        sample = np.random.multivariate_normal(self.gp_mu, self.gp_cov, (sample_size, self.latent_dim*2))
        sample = torch.from_numpy(np.float32(sample)).cuda()
        sample = sample.permute(0,2,1).contiguous()
        return sample

    def make_index2(self, index_list):
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

        return torch.LongTensor(index_1212).cuda(), torch.LongTensor(index_1122).cuda(), index_repeated

    ## Forward Pass
    def forward(self, condition, index_div):
        condition = condition.view(-1, 2)
        displace = self.cond_emb(condition)
        displace_back = self.cond_emb_back(condition)
        # displace = torch.cat((displace, torch.zeros(self.batch_size, 2).cuda()), dim=1)
        # displace = displace.view(self.batch_size,self.traj_num*2)
        # z = self.wiener_sample()
        z = self.gp_sample(condition.shape[0])
        z = z.contiguous().view(-1, self.out_length, self.latent_dim * 2).permute(1, 0, 2)

        scene_emb = self.ip_emb_g(z)
        scene_enc, _ = self.enc_lstm(scene_emb, h=displace.unsqueeze(0))

        scene_emb_back = torch.flip(scene_emb, dims=[0])
        scene_enc_back, _ = self.enc_lstm_back(scene_emb_back, h=displace_back.unsqueeze(0))
        scene_enc_back = torch.flip(scene_enc_back, dims=[0])

        scene_enc_1 = (scene_enc + scene_enc_back) / 2
        sequential_enc = self.sequential_mlp(scene_enc_1)

        index_1212, index_1122, index_repeated = self.make_index2(index_div)
        for id, idx in enumerate(index_repeated):
            if id == 0:
                idx_min = 0
            else:
                idx_min = index_repeated[id-1][-1]+1
            idx_max = idx_min + len(idx) - math.sqrt(len(idx))
            idx = np.arange(idx_min, idx_max).astype(int)
            index_repeated[id] = idx.tolist()
        z_1122 = sequential_enc[:, index_1122, :]
        z_1212 = sequential_enc[:, index_1212, :]
        z_relative = z_1122 - z_1212
        # 对于浮点数而言，可能存在相同的数相减之后加起来也不为0，精度问题。
        relative_index = torch.nonzero(index_1122 - index_1212, as_tuple=False)
        relative_index = relative_index.squeeze(1)
        z_relative = z_relative[:, relative_index, :]
        z_relative_enc = self.mlp(z_relative)
        z_relative_enc_ave = torch.cat([z_relative_enc[:,index, :].
                                        view(self.out_length,len(index_div[id]), len(index_div[id]) - 1, -1).mean(2)
                                         for id, index in enumerate(index_repeated)], dim=1)
        full_enc = torch.cat((sequential_enc, z_relative_enc_ave), dim=2)
        
        z_out = self.tanh(self.op_g(full_enc))
        z_out = z_out.view(self.out_length, -1, 2).permute(1, 2, 0)
        # sio.savemat('encoder_g.mat',{'scene_enc':scene_enc.detach().cpu().numpy(),
        #                              'scene_enc_back':scene_enc_back.detach().cpu().numpy(),
        #                              'sequential_enc':sequential_enc.detach().cpu().numpy(),
        #                              'condition':condition.detach().cpu().numpy(),
        #                              'index_div':index_div,
        #                              'z_out':z_out.contiguous().view(-1,2*self.out_length).detach().cpu().numpy()})
        # print('saved!')
        # time.sleep(5.0)
        # z_out = z_out.contiguous().view(self.batch_size, self.traj_num * 2 * self.out_length)
        # version 11.19
        # version before 11.19
        # z = z.view(self.batch_size,self.out_length,self.traj_num,2)
        # z = z.permute(1, 0, 2, 3).contiguous().view(self.out_length,-1,2)
        #
        # condition = condition.view(-1,2)
        # hero_pos_index = torch.where(torch.sum(abs(condition), dim=1) == 0)[0]
        # nbrs_pos_index = torch.nonzero(torch.sum(abs(condition), dim=1), as_tuple=False)
        # nbrs_pos_index = nbrs_pos_index.squeeze(1)
        # assert hero_pos_index.size(0) == self.batch_size
        # assert hero_pos_index.size(0) * (self.traj_num - 1) == nbrs_pos_index.size(0)
        # heroes = z[:, hero_pos_index, :]
        # heroes_cp = heroes.repeat(1, 1, self.traj_num - 1)
        # nbrs = z[:, nbrs_pos_index, :].view(self.out_length,self.batch_size,2*self.traj_num-2)
        # scene_relative = heroes_cp - nbrs
        # scene_relative_v = self.diff(scene_relative.view(self.out_length,-1,2), 1)
        # scene_relative_norm = scene_relative_v[:,:,0]**2+scene_relative_v[:,:,1]**2
        # scene_relative_theta = torch.atan2(scene_relative_v[:, :, 1], scene_relative_v[:, :, 0]+1e-6) / np.pi
        # scene_relative_vtheta = torch.stack((scene_relative_norm, scene_relative_theta), dim=2)
        #
        # hero_cond = condition[hero_pos_index, :]
        # hero_cond_cp = hero_cond.repeat(1, self.traj_num - 1)
        # nbr_cond = condition[nbrs_pos_index, :].view(self.batch_size, -1)
        # cond_relative = hero_cond_cp - nbr_cond
        #
        # cond_relative_enc = self.cond_emb(cond_relative)
        # scene_rela_conditioned_emb = self.op_spatial(scene_relative_vtheta.view(self.out_length,self.batch_size,-1))
        # scene_rela_conditioned_enc, _ = self.enc_lstm_spatial(scene_rela_conditioned_emb,
        #                                                       h=cond_relative_enc.unsqueeze(0))
        #
        # scene_emb = self.ip_emb_g(heroes)
        # scene_enc, _ = self.enc_lstm(scene_emb)
        #
        # # scene_emb_back = torch.flip(scene_emb, dims=[0])
        # # scene_enc_back, _ = self.enc_lstm_back(scene_emb_back)
        # # scene_enc_back = torch.flip(scene_enc_back, dims=[0])
        #
        # # scene_enc = torch.cat((scene_enc, scene_enc_back, scene_rela_conditioned_enc), dim=2)
        # scene_enc = torch.cat((scene_enc, scene_rela_conditioned_enc), dim=2)
        #
        # z_out = self.tanh(self.op_g(scene_enc))
        # z_out = z_out.view(self.out_length,self.batch_size,self.traj_num,2).permute(1,2,3,0)
        # z_out = z_out.contiguous().view(self.batch_size,self.traj_num*2*self.out_length)
        return z_out


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
        self.all_length = args['all_length']
        self.batch_size = args['batch_size']
        self.grid_size = args['grid_size']
        self.soc_conv_depth = args['soc_conv_depth']
        self.conv_3x1_depth = args['conv_3x1_depth']
        self.dyn_embedding_size = args['dyn_embedding_size']
        self.input_embedding_size = args['input_embedding_size']
        self.num_lat_classes = args['num_lat_classes']
        self.num_lon_classes = args['num_lon_classes']
        self.soc_embedding_size = (((args['grid_size'][0]-4)+1)//2)*self.conv_3x1_depth
        self.class_num = args['class_num']
        # self.soc_embedding_size = self.encoder_size

        ## Define network weights
        self.mlp_net = torch.nn.Sequential(
            torch.nn.Linear(2*self.in_length,self.input_embedding_size),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(self.input_embedding_size,self.encoder_size),
            torch.nn.LeakyReLU(0.1),
        )
        self.mlp_net_fut = torch.nn.Sequential(
            torch.nn.Linear(2 * self.out_length, self.input_embedding_size),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(self.input_embedding_size, self.encoder_size),
            torch.nn.LeakyReLU(0.1),
        )
        # Input embedding layer
        # self.ip_emb = torch.nn.Linear(2,self.input_embedding_size)

        # Encoder LSTM
        # self.enc_lstm = torch.nn.LSTM(self.input_embedding_size,self.encoder_size,1)

        # Vehicle dynamics embedding
        self.dyn_emb = torch.nn.Linear(self.encoder_size,self.dyn_embedding_size)
        self.nbr_emb = torch.nn.Linear(2*self.in_length,self.encoder_size)

        # Convolutional social pooling layer and social embedding layer
        # self.soc_conv = torch.nn.Conv2d(self.encoder_size,self.soc_conv_depth,3)
        # self.conv_3x1 = torch.nn.Conv2d(self.soc_conv_depth, self.conv_3x1_depth, (3,1))
        # self.soc_maxpool = torch.nn.MaxPool2d((2,1),padding = (1,0))

        # FC social pooling layer (for comparison):
        # self.soc_fc = torch.nn.Linear(self.soc_conv_depth * self.grid_size[0] * self.grid_size[1], (((args['grid_size'][0]-4)+1)//2)*self.conv_3x1_depth)

        # Decoder LSTM
        if self.use_maneuvers:
            # self.dec_lstm = torch.nn.LSTM(self.soc_embedding_size + self.dyn_embedding_size + self.num_lat_classes + self.num_lon_classes, self.decoder_size)
            self.out_linear = torch.nn.Linear(self.dyn_embedding_size + self.dyn_embedding_size + self.num_lat_classes + self.num_lon_classes, self.decoder_size)
        else:
            # self.dec_lstm = torch.nn.LSTM(self.soc_embedding_size + self.dyn_embedding_size, self.decoder_size)
            self.out_linear = torch.nn.Linear(3*self.dyn_embedding_size, self.decoder_size)

        # Output layers:
        self.op = torch.nn.Sequential(
            torch.nn.Linear(self.decoder_size,1),
            torch.nn.Sigmoid()
        )

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

    def make_index2(self, index_list):
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

        return torch.LongTensor(index_1212).cuda(), torch.LongTensor(index_1122).cuda(), index_repeated

    def forward(self, scene, hero_index,nbr_index, index_division):
        t_h = self.in_length-1
        t_f = self.out_length
        d_s = 1
        scene = scene.view(-1, 2, self.all_length)
        hero_traj = scene[hero_index, :, :]
        ref_pose = hero_traj[:, :, t_h].unsqueeze(2)

        # scene = scene.view(self.batch_size, self.traj_num, 2, self.all_length)
        index_batch = np.arange(0,self.batch_size)
        len_list = [len(i) for i in index_division]
        index_batch_rep = np.repeat(index_batch,len_list)
        ref_pose = ref_pose[index_batch_rep,:,:]

        scene = scene - ref_pose
        scene_scale = torch.cat([abs(scene[id, :, 0:t_h + 1:d_s]).max(2)[0].max(0)[0].unsqueeze(0).unsqueeze(2)
                                for id in index_division],dim=0)
        scene_scale = scene_scale[index_batch_rep,:]
        scene_scaled = scene / scene_scale

        # scene_scaled = scene_scaled.view(-1, 2, self.all_length)
        hero_traj = scene_scaled[hero_index, :, :]
        nbr_traj = scene_scaled[nbr_index, :, :]
        hist = hero_traj[:, :, 0:t_h+1:d_s].permute(2, 0, 1)
        fut = hero_traj[:, :, t_h+d_s:t_h + t_f+d_s:d_s].permute(2, 0, 1)
        nbrs = nbr_traj[:, :, 0:t_h+1:d_s].permute(2, 0, 1)
        for id, idx in enumerate(index_division):
            idx = np.array(idx) - id
            idx = np.arange(idx[0], idx[-1])
            index_division[id] = idx.tolist()

        hero_index = np.arange(hist.shape[1]).tolist()
        index_len = [len(i) for i in index_division]
        hero_repeated = np.repeat(hero_index, index_len)

        hist_enc = self.mlp_net(hist.permute(1,0,2).contiguous().view(-1,2*self.in_length))

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
        fut_enc = self.mlp_net_fut(fut.permute(1, 0, 2).contiguous().view(-1, 2 * self.out_length))
        fut_enc = self.leaky_relu(self.dyn_emb(fut_enc))

        enc = torch.cat((hist_enc,scene_pooled,fut_enc),1)
        # sio.savemat('rd_encoder.mat',{'hist_enc':hist_enc.detach().cpu().numpy(),
        #                               'scene_pooled':scene_pooled.detach().cpu().numpy(),
        #                               'fut_enc':fut_enc.detach().cpu().numpy(),
        #                               'index_div':index_division})
        # print('rd saved!')
        # enc = hist_enc

        feature_enc = self.leaky_relu(self.out_linear(enc))
        logit = self.op(feature_enc)
        return logit,feature_enc

    def decode(self,enc):
        h_dec = self.out_linear(enc)
        # print('after dec', h_dec.shape)
        fut_pred = self.op(h_dec)
        fut_pred = fut_pred.view(-1,2,self.out_length).permute(2, 0, 1)
        fut_pred = outputActivation(fut_pred)
        return fut_pred

def outputActivation(x):
    muX = x[:,:,0:1]
    muY = x[:,:,1:2]
    sigX = x[:,:,2:3]
    sigY = x[:,:,3:4]
    rho = x[:,:,4:5]
    sigX = torch.exp(sigX)
    sigY = torch.exp(sigY)
    rho = torch.tanh(rho)
    out = torch.cat([muX, muY, sigX, sigY, rho],dim=2)
    return out
