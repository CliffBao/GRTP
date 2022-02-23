from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
import scipy.io as scp
import numpy as np
import torch

#___________________________________________________________________________________________________________________________

### Dataset class for the NGSIM dataset
class ngsimDataset(Dataset):

    def __init__(self, mat_file, t_h=30, t_f=50, d_s=2, enc_size = 64, grid_size = (13,3)):
        self.D = scp.loadmat(mat_file)['traj']
        self.T = scp.loadmat(mat_file)['tracks']
        self.t_h = t_h  # length of track history
        self.t_f = t_f  # length of predicted trajectory
        self.d_s = d_s  # down sampling rate of all sequences
        self.enc_size = enc_size # size of encoder LSTM
        self.grid_size = grid_size # size of social context grid
        self.mat_file = mat_file
        if 'highd' in self.mat_file:
            self.t_h = 75
            self.t_f = 125
            self.d_s = 5
        # self.pre_processing()


    def __len__(self):
        return len(self.D)

    def pre_processing(self):
        invalid_idx = []
        for i in range(len(self.D)):
            print('processing %d / %d' % (i, len(self.D)))
            # first deal with neighbor
            grid = self.D[i, 8:]
            # then we should check whether invalid neighbor exists
            t = self.D[i, 2]
            hero_id = self.D[i, 1].astype(int)
            dsId = self.D[i, 0].astype(int)
            fut = self.getHistory(hero_id, t, hero_id, dsId)
            if fut.shape[0] == 0:
                invalid_idx.append(i)
                continue

            fut_max = abs(fut).max(axis=0)
            if np.sum(np.isnan(fut / fut_max)):
                invalid_idx.append(i)
                continue

            for idx, neighbor_id in enumerate(grid):
                if neighbor_id == 0:
                    continue
                if neighbor_id > self.T.shape[1]:
                    grid[idx] = 0
                    continue
                fut = self.getHistory(neighbor_id.astype(int), t, hero_id, dsId)
                if fut.shape[0] == 0:
                    invalid_idx.append(i)
                    continue

                fut_max = abs(fut).max(axis=0)
                if np.sum(np.isnan(fut / fut_max)):
                    invalid_idx.append(i)
                    continue
            self.D[i, 8:] = grid
            if not grid.nonzero()[0].shape[0]:
                invalid_idx.append(i)

        print('before processing, D shape is:', self.D.shape)
        self.D = np.delete(self.D, invalid_idx, axis=0)
        print('after processing, D shape is:', self.D.shape)
        scp.savemat(self.mat_file, {'traj': self.D, 'tracks': self.T})

    def __getitem__(self, idx):

        dsId = self.D[idx, 0].astype(int)
        vehId = self.D[idx, 1].astype(int)
        # t = self.D[idx, 2]
        # remember we use different now_time in train and test
        t = self.D[idx, 2]
        grid = self.D[idx,8:]
        neighbors = []

        # Get track history 'hist' = ndarray, and future track 'fut' = ndarray
        hist = self.getHistory(vehId,t,vehId,dsId)
        fut = self.getFuture(vehId,t,dsId)
        hero = hist

        # Get track histories of all neighbours 'neighbors' = [ndarray,[],ndarray,ndarray]
        # the coordinate is relative to this idx vehicle
        for i in grid:
            other = self.getHistory(i.astype(int), t,vehId,dsId)
            neighbors.append(other)
            if other.shape[0]>0:
                hero = np.concatenate((hero,other),axis=1)

        assert hero.shape[1]>0
        hero_max = abs(hero).max(axis=0)
        neigh_num = hero.shape[1]//2-1
        x_index = np.linspace(0, 2 * neigh_num, neigh_num + 1).astype(int)
        y_index = np.linspace(1, 2 * neigh_num + 1, neigh_num + 1).astype(int)
        x_max = hero_max[x_index].max()
        y_max = hero_max[y_index].max()
        xy_max = np.array([[x_max, y_max]])
        hero_max = np.repeat(xy_max, neigh_num + 1, axis=0).reshape(1, -1)

        # hero = (hero - hero_min) / (hero_max - hero_min) * 2 - 1
        hero = hero / hero_max
        hist = hist/hero_max[:,0:2]
        fut = fut/hero_max[:,0:2]
        for id,nei in enumerate(neighbors):
            if nei.shape[0]>0:
                # neighbors[id] = nei
                neighbors[id] = nei/hero_max[:,0:2]

        return hist,fut,neighbors,hero_max[:,0:2]

    # get 3s history trajectory from time t and convert to relative pos to refVehicle
    def getHistory(self,vehId,t,refVehId,dsId):
        if vehId == 0:
            return np.empty([0,2])
        else:
            if self.T.shape[1]<=vehId-1:
                return np.empty([0,2])
            refTrack = self.T[dsId-1][refVehId-1].transpose()
            vehTrack = self.T[dsId-1][vehId-1].transpose()

            refPos = refTrack[np.where(refTrack[:,0]==t)][0,1:3]
            if 'inter' in self.mat_file:
                theta = np.deg2rad(refTrack[np.where(refTrack[:,0]==t)][0,5]-90.0)
                assert(theta!=0)
                rotate_matrix = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
            if vehTrack.size==0 or np.argwhere(vehTrack[:, 0] == t).size==0:
                 return np.empty([0,2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s,1:3]-refPos
                if 'inter' in self.mat_file:
                    hist = np.dot(hist,rotate_matrix)
            if len(hist) < self.t_h//self.d_s + 1:
                return np.empty([0,2])
            return hist

    # get all future trajectory from time t and convert to relative pos to self
    def getFuture(self, vehId, t,dsId):
        vehTrack = self.T[dsId-1][vehId-1].transpose()
        refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3]
        if 'inter' in self.mat_file:
            theta = np.deg2rad(vehTrack[np.where(vehTrack[:,0]==t)][0,5]-90.0)
            assert(theta!=0)
            rotate_matrix = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s
        enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
        fut = vehTrack[stpt:enpt:self.d_s,1:3]-refPos
        if 'inter' in self.mat_file:
            fut = np.dot(fut,rotate_matrix)
        return fut

    ## Collate function for dataloader
    # for each time read, form a 25*batchsize*(x,y) dimension array
    # represent batchsize number of item, each item form a future 25 sample time
    # 简单来说，构成一个三维数组，每一行是第0维时候的未来位置，所以第0维是25
    # 第一维是batchsize，是把多条记录放在了一起
    # 所以这是批处理而不是实时计算。
    def collate_fn(self, samples):
        # Initialize neighbors and neighbors length batches:
        nbr_batch_size = 0
        for _,_,nbrs,_ in samples:
            nbr_batch_size += sum([len(nbrs[i])!=0 for i in range(len(nbrs))])
        maxlen = self.t_h//self.d_s + 1
        nbrs_batch = torch.zeros(maxlen,nbr_batch_size,2)

        # Initialize social mask batch:
        pos = [0, 0]
        index_division = [None] * len(samples)

        # Initialize history, history lengths, future, output mask, lateral maneuver and longitudinal maneuver batches:
        hist_batch = torch.zeros(maxlen,len(samples),2)
        fut_batch = torch.zeros(self.t_f//self.d_s,len(samples),2)
        scale_batch = []

        count = 0
        for sampleId,(hist, fut, nbrs, scale) in enumerate(samples):
            cur_count = 0
            # Set up history, future, lateral maneuver and longitudinal maneuver batches:
            hist_batch[0:len(hist),sampleId,0] = torch.from_numpy(hist[:, 0])
            hist_batch[0:len(hist), sampleId, 1] = torch.from_numpy(hist[:, 1])
            fut_batch[0:len(fut), sampleId, 0] = torch.from_numpy(fut[:, 0])
            fut_batch[0:len(fut), sampleId, 1] = torch.from_numpy(fut[:, 1])

            index = []
            # Set up neighbor, neighbor sequence length, and mask batches:
            for id,nbr in enumerate(nbrs):
                if len(nbr)!=0:
                    nbrs_batch[0:len(nbr),count,0] = torch.from_numpy(nbr[:, 0])
                    nbrs_batch[0:len(nbr), count, 1] = torch.from_numpy(nbr[:, 1])
                    pos[0] = id % self.grid_size[0]
                    pos[1] = id // self.grid_size[0]
                    index.append(count)
                    count += 1
                    cur_count += 1

            index_division[sampleId] = index
            # mask_batch[sampleId,:,:,:] = torch.from_numpy(grid)
            scale_batch.append(torch.from_numpy(scale))
        scale_batch = torch.cat(scale_batch,dim=0)

        return hist_batch, nbrs_batch, fut_batch, scale_batch,index_division

#________________________________________________________________________________________________________________________________________

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


## Batchwise MSE loss, uses mask for variable output lengths
def maskedMSE(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:,:,0]
    muY = y_pred[:,:,1]
    x = y_gt[:,:, 0]
    y = y_gt[:,:, 1]
    out = torch.pow(x-muX, 2) + torch.pow(y-muY, 2)
    acc[:,:,0] = out
    acc[:,:,1] = out
    acc = acc*mask
    lossVal = torch.sum(acc)/torch.sum(mask)
    return lossVal


# MSE loss for complete sequence, outputs a sequence of MSE values,
# uses mask for variable output lengths, used for evaluation
def maskedMSETest(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = torch.sum(acc[:,:,0],dim=1)
    counts = torch.sum(mask[:,:,0],dim=1)
    return lossVal, counts