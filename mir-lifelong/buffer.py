from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import numpy as np
import torch.nn as nn
import torch
from utils import maskedMSE
#___________________________________________________________________________________________________________________________

### Dataset class for the NGSIM dataset
class Buffer(nn.Module):

    def __init__(self, buffer_size):
        # time*num*2
        self.hist = None
        self.fut = None
        self.nbrs = []
        self.mem_size = buffer_size
        self.n_seen = 0

    def save_buffer(self,mat_file):
        index_start = []
        nbrs_len = []
        for i in range(len(self.nbrs)):
            if not index_start:
                index_start.append(0)
            else:
                index_start.append(nbrs_len[-1]+index_start[-1])
            nbrs_len.append(self.nbrs[i].shape[1])
        nbrs = np.concatenate(self.nbrs,axis=1)
        sio.savemat(mat_file,{'hist':self.hist,
                              'fut':self.fut,
                              'nbrs':nbrs,
                              'index_start':index_start,
                              'nbrs_len':nbrs_len,
                              'mem_size':self.mem_size,
                              'n_seen':self.n_seen})
        return

    def load_buffer(self,mat_file):
        data = sio.loadmat(mat_file)
        self.mem_size = data['mem_size']
        self.n_seen = data['n_seen']
        self.hist = data['hist']
        self.fut = data['hist']
        # nbrs: list of array
        self.nbrs = []
        nbrs = data['nbrs']
        index_start = data['index_start']
        nbrs_len = data['nbrs_len']
        for i in range(len(index_start)):
            self.nbrs.append(nbrs[:,index_start[i]:index_start[i]+nbrs_len[i],:])
        return

    def __len__(self):
        return self.hist.shape[1]

    def fetch_by_index(self,index_fetch):
        index_div = []
        hist = self.hist[:,index_fetch,:]
        fut = self.fut[:,index_fetch,:]
        nbrs = []
        for i in range(len(index_fetch)):
            cur = self.nbrs[index_fetch[i]]
            cur_len = cur.shape[1]
            if not index_div:
                index_div.append(list(range(cur_len)))
            else:
                index_div.append(list(range(index_div[-1][-1] + 1, index_div[-1][-1] + 1 + cur_len)))
            nbrs.append(cur)
        nbrs = np.concatenate(nbrs,axis=1)
        return torch.from_numpy(hist),torch.from_numpy(fut),torch.from_numpy(nbrs),index_div

    # 第一个任务后再使用buff进行retrieve，所以保证里面有数据而且远多于所需
    def mir_retrieve(self,model,old_model_param,retri_size):        
        if self.hist is None:
            return [],[],[],[]
        if self.__len__() < 2*retri_size:
            index_fetch = list(range(retri_size))
            hist, fut, nbrs, index_div = self.fetch_by_index(index_fetch)
        else:            
            index_permute = np.random.permutation(self.__len__())

            index_fetch = index_permute[0:2*retri_size]

            hist,fut,nbrs,index_div = self.fetch_by_index(index_fetch)            

            post_fut = model(hist.cuda(),nbrs.cuda(),index_div)
            post_rmse = torch.sqrt(torch.sum(torch.pow(post_fut-fut.cuda(),2),2)).mean(0)
            
            model.load_state_dict(old_model_param)
            pre_fut = model(hist.cuda(),nbrs.cuda(),index_div)
            pre_rmse = torch.sqrt(torch.sum(torch.pow(pre_fut-fut.cuda(),2),2)).mean(0)

            change = post_rmse-pre_rmse
            change_index = list(range(len(change)))
            change_sorted = sorted(zip(change,change_index),reverse=True)

            max_changed, max_changed_index = zip(*change_sorted[0:retri_size])

            index_fetch = index_fetch[list(max_changed_index)]
            hist,fut,nbrs,index_div = self.fetch_by_index(index_fetch)

            model.load_state_dict(old_model_param)
        return hist,nbrs,fut,index_div

    def reservior_update(self,hist,fut,nbrs,index_div):
        if self.hist is None:
            self.hist = hist.detach().cpu().numpy()
            self.fut = fut.detach().cpu().numpy()
            for index in index_div:
                self.nbrs.append(nbrs[:,index,:].detach().cpu().numpy())
            self.n_seen += hist.size(1)
            return []
        if self.hist.shape[1] < self.mem_size:
            self.hist = np.concatenate((self.hist,hist.detach().cpu().numpy()),axis=1)
            self.fut = np.concatenate((self.fut,fut.detach().cpu().numpy()),axis=1)
            for index in index_div:
                self.nbrs.append(nbrs[:, index, :].detach().cpu().numpy())
            self.n_seen += hist.size(1)
            return []
        # buffer满了，需要随机更新
        # 每个新加入数据的索引位置都生成一个随机数
        indices = torch.FloatTensor(hist.shape[1]).cuda().uniform_(0, self.n_seen).long()
        # 如果随机数小于buffer里面已经有的数据的大小
        valid_indices = (indices < self.hist.shape[1]).long()
        # 那么是新数据中哪些位置的可以用来更新
        idx_new_data = valid_indices.nonzero(as_tuple=False).squeeze(-1)
        # 用来更新buffer中哪些位置的数据
        idx_buffer = indices[idx_new_data]
        # 已经看过了多少数据
        self.n_seen += hist.shape[1]
        # 如果没有更新buffer中任何位置，返回空的更新位置列表
        if idx_buffer.numel() == 0:
            return []

        # overwrite
        for i in range(len(idx_new_data)):
            self.hist[:,idx_buffer[i],:] = hist[:,idx_new_data[i],:].detach().cpu().numpy()
            self.fut[:,idx_buffer[i],:] = fut[:,idx_new_data[i],:].detach().cpu().numpy()
            self.nbrs[idx_buffer[i]] = nbrs[:,index_div[idx_new_data[i]],:].detach().cpu().numpy()

        return idx_buffer


## Custom activation for output layer (Graves, 2015)
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


