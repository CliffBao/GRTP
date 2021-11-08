from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
import scipy.io as scp
import numpy as np
import torch
import math
import matplotlib.pyplot as plt

# ___________________________________________________________________________________________________________________________

### Dataset class for the NGSIM dataset
class ngsimDataset(Dataset):

    def __init__(self, mat_file, t_h=30, t_f=50, d_s=2, neigh_num=-1, enc_size=64, grid_size=(13, 3), hero_grid_pos=19,
                 length=80, use_cuda=True, local_normalise=0, batch_size=512, cond_length=0):
        self.D = scp.loadmat(mat_file)['traj']
        self.T = scp.loadmat(mat_file)['tracks']
        self.t_h = t_h  # length of track history
        self.t_f = t_f  # length of predicted trajectory
        self.d_s = d_s  # down sampling rate of all sequences
        self.enc_size = enc_size  # size of encoder LSTM
        self.grid_size = grid_size  # size of social context grid
        self.grid_scale = np.array([grid_size[0] - 1, grid_size[1] - 1])
        self.batch_size = batch_size
        self.cond_length = cond_length
        self.length = length
        self.use_cuda = use_cuda
        self.hero_grid_pos = hero_grid_pos
        self.local_normalise = local_normalise
        self.mat_file = mat_file
        self.neigh_num = neigh_num
        if 'highd' in self.mat_file:
            self.t_h = 75
            self.t_f = 125
            self.length = 200
            self.d_s = 5

        # self.setManeuver()
        # self.pre_processing()

    def pre_processing(self):
        invalid_idx = []
        neigh_num = []
        for i in range(len(self.D)):
            print('processing %d / %d' % (i, len(self.D)))
            # first deal with neighbor
            grid = self.D[i, 8:]
            grid[self.hero_grid_pos] = 0
            # then we should check whether invalid neighbor exists
            t = self.D[i, 2]
            hero_id = self.D[i, 1].astype(int)
            dsId = self.D[i, 0].astype(int)
            fut, is_valid, _ = self.getHistoryEM(hero_id, t, dsId)
            if not is_valid:
                invalid_idx.append(i)
                continue
            ref_index = 0
            ref_pos = fut[ref_index, :].reshape(1, 2)
            fut = fut - ref_pos
            fut_max = abs(fut).max(axis=0)
            if np.sum(np.isnan(fut / fut_max)):
                invalid_idx.append(i)
                continue

            for idx, neighbor_id in enumerate(grid):
                if neighbor_id > self.T.shape[1]:
                    grid[idx] = 0
                    continue
                fut, is_valid, _ = self.getHistoryEM(neighbor_id.astype(int), t, dsId)
                if not is_valid:
                    grid[idx] = 0
                    continue
                ref_pos = fut[ref_index, :].reshape(1, 2)
                fut = fut - ref_pos
                fut_max = abs(fut).max(axis=0)
                if np.sum(np.isnan(fut / fut_max)):
                    invalid_idx.append(i)
                    continue
            self.D[i, 8:] = grid
            if not grid.nonzero()[0].shape[0]:
                invalid_idx.append(i)
            else:
                neigh_num.append(grid.nonzero()[0].shape[0])

        print('before processing, D shape is:', self.D.shape)
        self.D = np.delete(self.D, invalid_idx, axis=0)
        print('after processing, D shape is:', self.D.shape)
        scp.savemat(self.mat_file, {'traj': self.D, 'tracks': self.T})
        nei_min = np.min(neigh_num)
        nei_max = np.max(neigh_num)
        for nei in range(nei_min, nei_max + 1):
            id = [index for index, val in enumerate(neigh_num) if val == nei]
            temp_d = self.D[id, :]
            file_name = 'data/TestSet-us1011-gan-smooth-nei' + str(nei) + '.mat'
            scp.savemat(file_name, {'traj': temp_d, 'tracks': self.T})
            print(file_name, ' saved! size: ', temp_d.shape)

    def setManeuver(self):
        for i in range(len(self.D)):
            print('setting %d / %d' % (i, len(self.D)))
            t = self.D[i, 2]
            vehId = self.D[i, 1].astype(int)
            dsId = self.D[i, 0].astype(int)

            vehTrack = self.T[dsId - 1][vehId - 1]
            # x maneuver is 1(hold lane),2(right change),3(left change)
            # we rescale them into [-1,1]
            # y maneuver is 1,2(decelerate)
            maneuver = self.D[i, 6:8] - np.array([2, 1])
            if vehTrack.shape[0] < 5:
                vehTrack = np.concatenate((vehTrack, np.zeros((2, vehTrack.shape[1]))), axis=0)
            stpt = np.argwhere(vehTrack[0, :] == t).item()

            vehTrack[3:5, stpt] = maneuver
            self.T[dsId - 1][vehId - 1] = vehTrack
        return

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):
        # here we assume that decomposed network can generate perfect single traj
        # so for generator, we need to generate several single trajs belongs to [-1,1]
        # for discriminator, we just sample dataset
        # mask and scene trajectories for discriminator, generator, generator_2
        dsId = self.D[idx, 0].astype(int)
        vehId = self.D[idx, 1].astype(int)
        t = self.D[idx, 2]
        grid = self.D[idx, 8:]
        neighbors = []

        hero, _, maneuver = self.getHistoryEM(vehId, t, dsId)
        hero_maneuver = maneuver.reshape((1, 2))
        ref_index = 0
        ref_pos = hero[ref_index, :].reshape(1, 2)
        # Get 8s traj of all neighbours 'neighbors' = [ndarray,[],ndarray,ndarray]
        neigh_num = 0
        valid_neigh_id = []
        maneuver_all = []
        for id, i in enumerate(grid):
            if (i > self.T.shape[1]) or (i.astype(int) == 0):
                nei = np.empty([0, 2])
            else:
                nei, _, maneuver = self.getHistoryEM(i.astype(int), t, dsId)
            neighbors.append(nei)
            maneuver_all.append(maneuver.reshape((1, 2)))
            if nei.shape[0] > 0:
                valid_neigh_id.append(id)
                neigh_num = neigh_num + 1
                other = nei
                hero = np.concatenate((hero, other), axis=1)

        ref_pos = np.repeat(ref_pos, neigh_num + 1, axis=0).reshape(1, 2 * neigh_num + 2)

        # Maneuvers 'lon_enc' = one-hot vector, 'lat_enc = one-hot vector
        # lon_enc = np.zeros([2])
        # lon_enc[int(self.D[idx, 7] - 1)] = 1
        # lat_enc = np.zeros([3])
        # lat_enc[int(self.D[idx, 6] - 1)] = 1

        # raw_cp = hero
        fake_item = hero
        fake_item = fake_item - fake_item[ref_index, :]
        fake_item_2 = hero
        fake_item_2 = fake_item_2 - fake_item_2[ref_index, :]
        hero = hero - ref_pos

        # here we assume that single traj is perfect generated
        # so we use real single traj to validate model

        if self.local_normalise:
            if hero.shape[1] == 2:
                hero_max = abs(hero).max(axis=0)
                hero = hero / hero_max
            else:

                hero_max = abs(hero).max(axis=0)
                x_index = np.linspace(0, 2 * neigh_num, neigh_num + 1).astype(int)
                y_index = np.linspace(1, 2 * neigh_num + 1, neigh_num + 1).astype(int)
                x_max = hero_max[x_index].max()
                y_max = hero_max[y_index].max()
                xy_max = np.array([[x_max, y_max]])
                hero_max = np.repeat(xy_max, neigh_num + 1, axis=0).reshape(1, -1)

                # hero = hero / hero_max
                hero_max = hero_max.reshape(-1, 2)

        else:
            hero = (hero - ref_pos) / self.scale_arr
        real_item = hero
        # then we get fake data according to conditions resulted from real data
        # shape: self.length*(2*self.neigh_num+2)
        # each column is a time sequence of x,y iteratively
        # x1,y1,x2,y2,x3,y3,...

        # fake sample is x1,x2,...,xn,y1,y2,...,yn
        # local normalise is not required for fake sample
        # if self.local_normalise:
        #     fake_max = abs(fake_item).max(axis=0)
        #     fake_item = fake_item / fake_max

        #     fake_max_2 = abs(fake_item_2).max(axis=0)
        #     fake_item_2 = fake_item_2 / fake_max_2

        hero_real = real_item[:, 0:2]
        hero_fake = fake_item[:, 0:2]
        hero_fake_2 = fake_item_2[:, 0:2]
        # use shallow copy, not deep copy(=)
        neigh_real = neighbors[:]
        neigh_fake = neighbors[:]
        neigh_fake_2 = neighbors[:]
        # print('real_item',real_item)
        # print('fake_item',fake_item)
        # print('fake_item2',fake_item_2)
        # assert 3>4
        for idx, i in enumerate(valid_neigh_id):
            neigh_real[i] = real_item[:, 2 * idx + 2:2 * idx + 4]
            neigh_fake[i] = fake_item[:, 2 * idx + 2:2 * idx + 4]
            neigh_fake_2[i] = fake_item_2[:, 2 * idx + 2:2 * idx + 4]
        # 3,13 grid, in origin should be  hero
        # we add hero into the scene
        # if neigh_real[self.hero_grid_pos].shape[0]>0:
        #     print(neigh_real[self.hero_grid_pos])
        # assert neigh_real[self.hero_grid_pos].shape[0]==0
        neigh_real[self.hero_grid_pos] = hero_real
        neigh_fake[self.hero_grid_pos] = hero_fake
        neigh_fake_2[self.hero_grid_pos] = hero_fake_2
        maneuver_all[self.hero_grid_pos] = hero_maneuver

        return neigh_real, neigh_fake, neigh_fake_2, maneuver_all, hero_max

    ## Helper function to get track history
    # get 3s history trajectory from time t and convert to relative pos to refVehicle
    def getHistory(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 2])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 2])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()

            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 1:3]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos

            if len(hist) < self.t_h // self.d_s:
                return np.empty([0, 2])
            return hist

    ## Helper function to get track future
    # get all future trajectory from time t and convert to relative pos to self
    def getHistoryEM(self, vehId, t, dsId):
        maneuver = np.zeros((1, 2))
        vehTrack = self.T[dsId - 1][vehId - 1].transpose()
        if (vehTrack.size == 0) or (np.argwhere(vehTrack[:, 0] == t).size == 0):
            return np.empty([0, 2]), False, maneuver
        # refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3]
        stpt = np.argwhere(vehTrack[:, 0] == t).item()-self.t_h
        enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f+1)
        if stpt >= enpt:
            return np.empty([2, 1]), False, maneuver
        # fut = vehTrack[stpt:enpt:self.d_s, 1:3]-refPos
        fut = vehTrack[stpt:enpt:self.d_s, 1:3]
        # fut = fut.reshape((1, 2))
        row_num = fut.shape[0]
        if row_num < self.length // self.d_s+1:
            # print('Warning: row num is not enough!!!')
            # fake_traj = np.repeat(fut[-1][:].reshape(1,2), int(self.t_f / self.d_s - row_num), axis=0)
            # fut = np.concatenate((fut, fake_traj), axis=0)
            fut = np.empty([0, 2])
            return fut, False, maneuver
        maneuver = vehTrack[stpt, 3:5]
        return fut, True, maneuver

    ## Collate function for dataloader
    # for each time read, form a 25*batchsize*(x,y) dimension array
    # represent batchsize number of item, each item form a future 25 sample time
    # 简单来说，构成一个三维数组，每一行是第0维时候的未来位置，所以第0维是25
    # 第一维是batchsize，是把多条记录放在了一起
    # 所以这是批处理而不是实时计算。
    def collate_fn(self, samples):

        # Initialize neighbors and neighbors length batches:
        nbr_batch_size = 0
        for nbrs, _, _, _, _ in samples:
            nbr_batch_size += sum([len(nbrs[i]) != 0 for i in range(len(nbrs))])
            # nbr_batch_size += (real.shape[1]-2)
        maxlen = self.length // self.d_s+1

        # Initialize social mask batch:
        pos = [0, 0]
        mask_batch = torch.zeros(len(samples), self.grid_size[1], self.grid_size[0], self.enc_size)
        mask_batch = mask_batch.bool()
        index_division = [None] * self.batch_size

        # Initialize history, history lengths, future, output mask, lateral maneuver and longitudinal maneuver batches:

        nbrs_real_batch = torch.zeros(maxlen, nbr_batch_size, 2)
        nbrs_fake_batch = torch.zeros(maxlen, nbr_batch_size, 2)
        nbrs_fake_2_batch = torch.zeros(maxlen, nbr_batch_size, 2)
        maneuver_batch = torch.zeros(nbr_batch_size, 2)
        scale_batch = []

        count_real = 0
        count_fake = 0
        count_fake_2 = 0
        for sampleId, (nbrs, nbrs_fake, nbrs_fake_2, maneuver, scale) in enumerate(samples):
            # Set up neighbor, neighbor sequence length, and mask batches:
            index = []

            scale_batch.append(torch.from_numpy(scale))
            for id, nbr in enumerate(nbrs):
                if len(nbr) != 0:
                    nbrs_real_batch[0:len(nbr), count_real, 0] = torch.from_numpy(nbr[:, 0])
                    nbrs_real_batch[0:len(nbr), count_real, 1] = torch.from_numpy(nbr[:, 1])
                    maneuver_batch[count_real, :] = torch.from_numpy(maneuver[id])
                    pos[0] = id % self.grid_size[0]
                    pos[1] = id // self.grid_size[0]
                    mask_batch[sampleId, pos[1], pos[0], :] = torch.ones(self.enc_size).byte()
                    index.append(count_real)
                    count_real += 1
            index_division[sampleId] = index
            for id, nbr in enumerate(nbrs_fake):
                if len(nbr) != 0:
                    nbrs_fake_batch[0:len(nbr), count_fake, 0] = torch.from_numpy(nbr[:, 0])
                    nbrs_fake_batch[0:len(nbr), count_fake, 1] = torch.from_numpy(nbr[:, 1])
                    count_fake += 1
            for id, nbr in enumerate(nbrs_fake_2):
                if len(nbr) != 0:
                    nbrs_fake_2_batch[0:len(nbr), count_fake_2, 0] = torch.from_numpy(nbr[:, 0])
                    nbrs_fake_2_batch[0:len(nbr), count_fake_2, 1] = torch.from_numpy(nbr[:, 1])
                    count_fake_2 += 1
        scale_batch = torch.cat(scale_batch, 0)
        return nbrs_real_batch, nbrs_fake_batch, nbrs_fake_2_batch, mask_batch, index_division, maneuver_batch, scale_batch


# ________________________________________________________________________________________________________________________________________


## Custom activation for output layer (Graves, 2015)
def outputActivation(x):
    muX = x[:, :, 0:1]
    muY = x[:, :, 1:2]
    sigX = x[:, :, 2:3]
    sigY = x[:, :, 3:4]
    rho = x[:, :, 4:5]
    sigX = torch.exp(sigX)
    sigY = torch.exp(sigY)
    rho = torch.tanh(rho)
    out = torch.cat([muX, muY, sigX, sigY, rho], dim=2)
    return out


## Batchwise NLL loss, uses mask for variable output lengths
def maskedNLL(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    sigX = y_pred[:, :, 2]
    sigY = y_pred[:, :, 3]
    rho = y_pred[:, :, 4]
    ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY,
                                                                                                           2) - 2 * rho * torch.pow(
        sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = torch.sum(acc) / torch.sum(mask)
    return lossVal


## NLL for sequence, outputs sequence of NLL values for each time-step, uses mask for variable output lengths, used for evaluation
def maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask, num_lat_classes=3, num_lon_classes=2, use_maneuvers=True,
                  avg_along_time=False):
    if use_maneuvers:
        acc = torch.zeros(op_mask.shape[0], op_mask.shape[1], num_lon_classes * num_lat_classes).cuda()
        count = 0
        for k in range(num_lon_classes):
            for l in range(num_lat_classes):
                wts = lat_pred[:, l] * lon_pred[:, k]
                wts = wts.repeat(len(fut_pred[0]), 1)
                y_pred = fut_pred[k * num_lat_classes + l]
                y_gt = fut
                muX = y_pred[:, :, 0]
                muY = y_pred[:, :, 1]
                sigX = y_pred[:, :, 2]
                sigY = y_pred[:, :, 3]
                rho = y_pred[:, :, 4]
                ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
                x = y_gt[:, :, 0]
                y = y_gt[:, :, 1]
                out = -(torch.pow(ohr, 2) * (
                            torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY,
                                                                                                        2) - 2 * rho * torch.pow(
                        sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr))
                acc[:, :, count] = out + torch.log(wts)
                count += 1
        acc = -logsumexp(acc, dim=2)
        acc = acc * op_mask[:, :, 0]
        if avg_along_time:
            lossVal = torch.sum(acc) / torch.sum(op_mask[:, :, 0])
            return lossVal
        else:
            lossVal = torch.sum(acc, dim=1)
            counts = torch.sum(op_mask[:, :, 0], dim=1)
            return lossVal, counts
    else:
        acc = torch.zeros(op_mask.shape[0], op_mask.shape[1], 1).cuda()
        y_pred = fut_pred
        y_gt = fut
        muX = y_pred[:, :, 0]
        muY = y_pred[:, :, 1]
        sigX = y_pred[:, :, 2]
        sigY = y_pred[:, :, 3]
        rho = y_pred[:, :, 4]
        ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
        x = y_gt[:, :, 0]
        y = y_gt[:, :, 1]
        out = torch.pow(ohr, 2) * (
                torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY,
                                                                                            2) - 2 * rho * torch.pow(
            sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr)
        acc[:, :, 0] = out
        acc = acc * op_mask[:, :, 0:1]
        if avg_along_time:
            lossVal = torch.sum(acc[:, :, 0]) / torch.sum(op_mask[:, :, 0])
            return lossVal
        else:
            lossVal = torch.sum(acc[:, :, 0], dim=1)
            counts = torch.sum(op_mask[:, :, 0], dim=1)
            return lossVal, counts


## Batchwise MSE loss, uses mask for variable output lengths
def maskedMSE(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = torch.sum(acc) / torch.sum(mask)
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
    lossVal = torch.sum(acc[:, :, 0], dim=1)
    counts = torch.sum(mask[:, :, 0], dim=1)
    return lossVal, counts


## Helper function for log sum exp calculation:
def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


# add mmd function to test distribution discrepancy during training
# 这种直接将（n+m）复制（n+m）次的方法虽然简单直接，但是会占用很大的内存
# 比如4000*162的轨迹序列，串接后变成8000*162，复制后变成64000000*162
# 直接将内存占用放大了16000，或者说4n倍，从而导致显存不够
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """计算Gram/核矩阵
    source: sample_size_1 * feature_size 的数据
    target: sample_size_2 * feature_size 的数据
    kernel_mul: 这个概念不太清楚，感觉也是为了计算每个核的bandwith
    kernel_num: 表示的是多核的数量
    fix_sigma: 表示是否使用固定的标准差

		return: (sample_size_1 + sample_size_2) * (sample_size_1 + sample_size_2)的
						矩阵，表达形式:
						[	K_ss K_st
							K_ts K_tt ]
    """
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)  # 合并在一起

    total0 = total.unsqueeze(0).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)  # 计算高斯核中的|x-y|

    # 计算多核中每个核的bandwidth
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    # 高斯核的公式，exp(-|x-y|/bandwith)
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
                  bandwidth_temp in bandwidth_list]

    return sum(kernel_val)  # 将多个核合并在一起


def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul,
                              kernel_num=kernel_num,
                              fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]  # Source<->Source
    YY = kernels[batch_size:, batch_size:]  # Target<->Target
    XY = kernels[:batch_size, batch_size:]  # Source<->Target
    YX = kernels[batch_size:, :batch_size]  # Target<->Source
    loss = torch.mean(XX + YY - XY - YX)  # 这里是假定X和Y的样本数量是相同的
    # 当不同的时候，就需要乘上上面的M矩阵
    return loss


# 重写mmd降低占用内存
def mmd_mine(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)  # 合并在一起
    #
    # total0 = total.unsqueeze(0).expand(int(total.size(0)), \
    #                                    int(total.size(0)), \
    #                                    int(total.size(1)))
    # total1 = total.unsqueeze(1).expand(int(total.size(0)), \
    #                                    int(total.size(0)), \
    #                                    int(total.size(1)))
    L2_distance = torch.cdist(total, total, p=2)  # 计算高斯核中的|x-y|

    # 计算多核中每个核的bandwidth
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    # 高斯核的公式，exp(-|x-y|/bandwith)
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
                  bandwidth_temp in bandwidth_list]

    kernels = sum(kernel_val)
    XX = kernels[:batch_size, :batch_size]  # Source<->Source
    YY = kernels[batch_size:, batch_size:]  # Target<->Target
    XY = kernels[:batch_size, batch_size:]  # Source<->Target
    YX = kernels[batch_size:, :batch_size]  # Target<->Source
    loss = torch.mean(XX + YY - XY - YX)  # 这里是假定X和Y的样本数量是相同的
    # 当不同的时候，就需要乘上上面的M矩阵
    return loss
