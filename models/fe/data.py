import os
from time import time
import math
import random
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset
import torch.distributed as td

from util import write_log, time_stat


class ProtStructDat(Dataset):

    def __init__(self, batch_size, queue_size, coordinate_fp_list, n_ref_points, use_cnn, logger):
        random.seed(999)
        self.bz, self.qz = batch_size, queue_size
        self.n_ref_points = n_ref_points
        self._xyz_dict_path_list = coordinate_fp_list
        self.use_cnn = use_cnn
        self.logger = logger

        self.samples = {}
        self.result_id_list = []
        self.node_fea_dict, self.adj_mat_dict = {}, {}
        self.pair_num_list = []

        write_log(logger, 'Extracting raw features...')
        self._extract_features_all()

    def __getitem__(self, index):
        return self._pack_input(self.samples[index])

    def _pack_input(self, sample):
        return (self.node_fea_dict[sample[0]], self.node_fea_dict[sample[1]],
                self.adj_mat_dict[sample[0]], self.adj_mat_dict[sample[1]])

    def get_data_path(self):
        raise NotImplementedError

    def get_annotation(self, logger):
        pair_dir_path = self.get_data_path()
        order_fp = os.path.join(pair_dir_path, 'pdb_order_list.txt')
        pair_file_list = []
        with open(order_fp, 'r') as order_f:
            for line in order_f:
                pair_file_list.append(line.strip())
        write_log(logger, 'reading annotation files...')

        for pair_fn in pair_file_list:
            # check if exists
            if pair_fn not in self.node_fea_dict.keys():
                continue
            pair_fp = os.path.join(pair_dir_path, pair_fn + '.pairs')
            self.read_pair_file(pair_fp)

    def read_pair_file(self, pair_fp):
        raise NotImplementedError

    def _extract_features_all(self):
        xyz_dict = {}
        for xyz_dict_path in self._xyz_dict_path_list:
            xyz_dict.update(dict(np.load(xyz_dict_path)))
        pdb_fn_list = xyz_dict.keys()
        for pdb_fn in pdb_fn_list:
            if self.use_cnn:
                fea, self.adj_mat_dict[pdb_fn] = self.extract_features_single_cnn(xyz_dict[pdb_fn])
            else:
                fea, self.adj_mat_dict[pdb_fn] = self.extract_features_single(xyz_dict[pdb_fn], self.n_ref_points)
            self.node_fea_dict[pdb_fn] = torch.tensor(fea, dtype=torch.float)

    @staticmethod
    def extract_features_single(xyz_list, n_ref_points):
        # get raw node features
        rxyz = ProtStructDat.get_relative_coordinate(xyz_list, n_ref_points)
        alpha_angle = ProtStructDat.cal_alphac_angle(xyz_list)
        fea = np.concatenate((rxyz, alpha_angle), axis=1)
        # get the adjacency matrix
        omega, epsilon = 4.0, 2.0
        p = np.array(xyz_list)
        vp = np.expand_dims(p, axis=1)
        dist_mat = np.sqrt(np.sum(np.square(vp - p), axis=2))
        adj_mat = np.divide(omega, np.maximum(dist_mat, epsilon))

        return fea, adj_mat

    @staticmethod
    def extract_features_single_cnn(xyz_list):
        p = np.array(xyz_list)
        vp = np.expand_dims(p, axis=1)
        dist_mat = np.sum(np.square(vp - p), axis=2)
        len_dm = dist_mat.shape[0]
        dist_mat[range(len_dm), range(len_dm)] = float('inf')
        inv_dm = np.divide(1.0, dist_mat)
        fea = np.zeros((len(xyz_list), 1))
        return fea, inv_dm


    @staticmethod
    def get_relative_coordinate(xyz, n_ref_points):
        """
        :param xyz: size is N x 3
        :param n_ref_points: number of reference points
        :return: relative position
        """
        xyz = np.array(xyz)
        group_num = int(np.log2(n_ref_points + 1))
        assert 2 ** group_num - 1 == n_ref_points,\
            "The number of anchor points is {} and should be 2^k - 1, " \
            "where k is an integer, but k is {}.".format(n_ref_points, group_num)
        n_points = xyz.shape[0]
        ref_points = []
        for i in range(group_num):
            n_points_in_group = 2 ** i
            for j in range(n_points_in_group):
                beg, end = n_points * j // n_points_in_group, math.ceil(n_points * (j + 1) / n_points_in_group)
                ref_point = np.mean(xyz[beg:end, :], axis=0)
                ref_points.append(ref_point)
        coordinates = [np.linalg.norm(xyz - rp, axis=1).reshape(-1, 1) for rp in ref_points]

        return np.concatenate(coordinates, axis=1)

    @staticmethod
    def cal_alphac_angle(xyz):
        direction_vec = np.array(xyz)[1:, :] - np.array(xyz)[:-1, :]
        dv_1 = direction_vec[:-1, :]
        dv_2 = direction_vec[1:, :]
        dv_dot = np.sum(dv_1 * dv_2, axis=1)
        dv_norm = np.linalg.norm(dv_1, axis=1) * np.linalg.norm(dv_2, axis=1)
        # padding for two terminals
        pad_dv_norm = np.zeros((len(xyz)))
        pad_dv_norm[1:-1] = dv_dot / dv_norm

        return pad_dv_norm.reshape((-1, 1))


class SCOPe(ProtStructDat):

    def __init__(self, batch_size, queue_size, coordinate_fp_list, n_ref_points, use_cnn, logger, fold_k):
        self.k = fold_k
        super().__init__(batch_size=batch_size, queue_size=queue_size, coordinate_fp_list=coordinate_fp_list,
                         n_ref_points=n_ref_points, use_cnn=use_cnn, logger=logger)


class SCOPeTrain(SCOPe):
    def __init__(self, batch_size, queue_size, coordinate_fp_list, n_ref_points, use_cnn, logger, fold_k, dyna_par):
        self.cur_epoch = 0
        self.dyna_par = dyna_par
        super().__init__(batch_size=batch_size, queue_size=queue_size, coordinate_fp_list=coordinate_fp_list,
                         n_ref_points=n_ref_points, use_cnn=use_cnn, logger=logger, fold_k=fold_k)

        if self.dyna_par:
            sorted_tr_samples_path = "./dataset/scope207/pair_list_for_train/tr_pair_list_{}_P3_B{}_Q{}".format(
                fold_k, batch_size, queue_size)
            write_log(logger, sorted_tr_samples_path)
            with open(sorted_tr_samples_path, 'rb') as stsf:
                self.sorted_tr_samples = pickle.load(stsf)
            pos_pair_dict_path = "./dataset/scope207/pair_list_for_train/tr_pair_{}.pkl".format(fold_k)
            if not os.path.exists(pos_pair_dict_path):
                self.tms_mat, self.sorted_pdb_fn_dict = {}, {}
                self.n_tr_prots = -1
                self.get_annotation(logger)
                pair_data = {
                    'tms_mat': self.tms_mat, 'sorted_pdb': self.sorted_pdb_fn_dict, 'n_tr_prots': self.n_tr_prots
                }
                with open(pos_pair_dict_path, 'wb') as ppdf:
                    pickle.dump(pair_data, ppdf)
            self.n_samples = 130000
        else:
            self.get_annotation(logger)
            self.n_samples = len(self.samples)

    def __len__(self):
        return self.n_samples // self.bz * self.bz

    def __getitem__(self, index):
        if index == td.get_rank():
            if self.dyna_par:
                self.result_id_list = self.sorted_tr_samples[self.cur_epoch % 120]
            else:
                self.sort_samples_moco(self.cur_epoch)
        sa_id = self.result_id_list[index]
        sa = (sa_id[0], sa_id[1])

        if index == self.__len__() - 1 or index == self.__len__() - 2:
            self.logger.info("one epoch ends!")
            self.cur_epoch += 1
        return self._pack_input(sa)

    def get_data_path(self):
        pair_dir_path = './dataset/scope207/train_{}_anno'.format(self.k)
        assert os.path.exists(pair_dir_path)
        write_log(self.logger, 'Reading training data...')
        return pair_dir_path

    def read_pair_file(self, pair_fp):
        tms_dict = {}
        pdb_fn1 = None
        # get positive pairs
        with open(pair_fp, 'r') as pair_f:
            for line in pair_f:
                pdb_fn1, pdb_fn2, tms, label = line.strip().split(',')
                if pdb_fn2 not in self.node_fea_dict.keys():
                    continue
                if label == '1' and tms == '1.0':
                    continue
                if self.dyna_par:
                    tms_dict[pdb_fn2] = float(tms)
                elif label == '1':
                    self.samples[pdb_fn1, pdb_fn2] = float(tms)

        if self.dyna_par:
            sorted_pdb_tms_pairs = sorted(tms_dict.items(), key=lambda x: x[1], reverse=True)
            sorted_pdb_fns = [x[0] for x in sorted_pdb_tms_pairs]
            if self.n_tr_prots == -1:
                self.n_tr_prots = len(sorted_pdb_fns)
            else:
                assert self.n_tr_prots == len(sorted_pdb_fns)
            self.sorted_pdb_fn_dict[pdb_fn1] = sorted_pdb_fns
            self.tms_mat[pdb_fn1] = tms_dict

    @time_stat
    def sort_samples_moco(self, rand_seed):
        self.result_id_list = []
        n_bz = self.n_samples // self.bz
        random.seed(rand_seed)
        keys_queue = []
        queue_len = self.qz // self.bz
        ori_id_list = list(self.samples.keys())
        for k in range(n_bz):
            new_key_pool = []
            single_batch_id_list = []
            for i in range(self.bz):
                while True:
                    chosen_sample = random.choice(ori_id_list)
                    find_flag = True
                    for keys_pool in keys_queue:
                        for key_pdb in keys_pool:
                            if (chosen_sample[0], key_pdb) in self.samples:
                                find_flag = False
                                break
                        if not find_flag:
                            break
                    if find_flag:
                        break
                single_batch_id_list.append(chosen_sample)
                new_key_pool.append(chosen_sample[1])
            self.result_id_list.extend(single_batch_id_list)
            keys_queue.append(new_key_pool)
            if len(keys_queue) > queue_len:
                keys_queue.pop(0)


class SCOPeTest(SCOPe):

    def __init__(self, batch_size, coordinate_fp_list, n_ref_points, use_cnn, logger, fold_k, which_set='Train'):
        self.__which_set = which_set
        self.read_tr_flag = True
        super().__init__(batch_size=batch_size, queue_size=-1, coordinate_fp_list=coordinate_fp_list,
                         n_ref_points=n_ref_points, use_cnn=use_cnn, logger=logger, fold_k=fold_k)
        self.labels = []
        self.train_samples, self.test_samples = [], []
        self.get_annotation(logger)
        self.n_tr_samples, self.n_te_samples = len(self.train_samples), len(self.test_samples)
        write_log(logger, "Number of samples in training and testing set {}: {}, {}".format(
            fold_k, self.n_tr_samples, self.n_te_samples))

    def __len__(self):
        if self.__which_set == 'Train':
            return self.n_tr_samples
        elif self.__which_set == 'Test':
            return self.n_te_samples
        else:
            raise RuntimeError('self.which_set = {} is invalid!'.format(self.__which_set))

    def __getitem__(self, index):
        if self.__which_set == 'Train':
            sa = self.train_samples[index]
        elif self.__which_set == 'Test':
            sa = self.test_samples[index]
        else:
            raise RuntimeError('self.which_set = {} is invalid!'.format(self.__which_set))
        return self._pack_input(sa)

    def get_data_path(self):
        pair_dir_path = './dataset/scope207/test_{}_anno'.format(self.k)
        assert os.path.exists(pair_dir_path)
        write_log(self.logger, 'Reading testing data...')
        return pair_dir_path

    def read_pair_file(self, pair_fp):
        pn, pos_num, neg_num = 0, 0, 0
        single_sample_labels = []
        with open(pair_fp, 'r') as pair_f:
            for line in pair_f:
                pdb_fn1, pdb_fn2, tms, label = line.strip().split(',')
                if pdb_fn2 not in self.node_fea_dict.keys():
                    continue
                single_sample_labels.append(float(label))
                if self.read_tr_flag:
                    self.train_samples.append((pdb_fn2, pdb_fn2))
                pn += 1
                if label == '1':
                    pos_num += 1
                else:
                    neg_num += 1
        if pos_num > 0:
            self.labels.extend(single_sample_labels)
            self.test_samples.append((pdb_fn1, pdb_fn1))
            self.pair_num_list.append(pn)
        self.read_tr_flag = False

    def select_set(self, set_name):
        self.__which_set = set_name

    def get_labels(self):
        return self.labels


class NewRelease(SCOPeTest):

    def __init__(self, batch_size, coordinate_fp_list, n_ref_points, use_cnn, logger, which_set='Train'):
        super().__init__(batch_size=batch_size, coordinate_fp_list=coordinate_fp_list, n_ref_points=n_ref_points,
                         use_cnn=use_cnn, logger=logger, fold_k=-1, which_set=which_set)

    def get_data_path(self):
        pair_dir_path = './dataset/new_release/new_anno'
        assert os.path.exists(pair_dir_path)
        write_log(self.logger, 'Reading testing data...')
        return pair_dir_path, 196
