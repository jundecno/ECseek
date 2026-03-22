import os
import random
import pickle
from time import *
import logging
from collections import Iterable

import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix
import torch
import torch.nn as nn
import torch.distributed as td


def show_dataset_statistics(xyz_dict_path):
    xyz_dict = np.load(xyz_dict_path)
    max_len, min_len = 0, 99999
    for pdb_fn in xyz_dict:
        seq_len = xyz_dict[pdb_fn].shape[0]
        if seq_len > max_len:
            max_len = seq_len
        if seq_len < min_len:
            min_len = seq_len
    print('max length:', max_len, 'min length:', min_len)

    for i in range(5):
        pos_num, neg_num = 0, 0
        anno_dir = 'dataset/train_{}_anno'.format(i)
        assert os.path.exists(anno_dir)
        anno_fn_list = os.listdir(anno_dir)
        for anno_fn in anno_fn_list:
            if anno_fn == 'pdb_order_list.txt':
                continue
            anno_fp = os.path.join(anno_dir, anno_fn)
            with open(anno_fp, 'r') as anno_f:
                for line in anno_f:
                    label = line.strip().split(',')[3]
                    if label == '1':
                        pos_num += 1
                    else:
                        neg_num += 1
        avg_pos_ratio = pos_num / (pos_num + neg_num)
        avg_neg_ratio = 1 - avg_pos_ratio
        print('Positive num: {} Positive ratio: {}'.format(pos_num, avg_pos_ratio))
        print('Negative num: {} Negative ratio: {}'.format(neg_num, avg_neg_ratio))


def eval_metric(pred_labels, real_labels, pair_num_data, verse=True):
    pred_num = len(pred_labels)
    assert pred_num == len(real_labels)
    if not isinstance(pair_num_data, Iterable):
        # print(pred_num, pair_num_data)
        assert pred_num % pair_num_data == 0
        pair_num_list = [pair_num_data for _ in range(round(pred_num / pair_num_data))]
    else:
        assert pred_num == sum(pair_num_data)
        pair_num_list = pair_num_data

    if verse:
        v_pred_labels = [-x for x in pred_labels]
        pred_labels_topk = pred_labels
    else:
        v_pred_labels = pred_labels
        pred_labels_topk = [-x for x in pred_labels]
    query_num = len(pair_num_list)

    roc_auc_list, prc_auc_list = [], []
    hit_1_num, hit_5_num, hit_10_num = 0, 0, 0
    hit_1_ratio, hit_5_ratio, hit_10_ratio = 0, 0, 0
    i = 0
    for j, pair_num in enumerate(pair_num_list):
        assert sum(real_labels[i: i + pair_num]) != 0
        try:
            fpr, tpr, _ = roc_curve(
                real_labels[i: i + pair_num], v_pred_labels[i: i + pair_num], pos_label=1, drop_intermediate=False)
        except ValueError:
            print(real_labels[i: i + pair_num])
            print(v_pred_labels[i: i + pair_num])
            assert False
        pre, rec, _ = precision_recall_curve(
            real_labels[i: i + pair_num], v_pred_labels[i: i + pair_num], pos_label=1)
        # rec, pre = cal_prc_curve(pred_labels, real_labels)
        roc_auc_list.append(auc(fpr, tpr))
        prc_auc_list.append(auc(rec, pre))
        hit_1_num += top_k_acc(pred_labels_topk[i: i + pair_num], real_labels[i: i + pair_num], 1)
        hit_5_num += top_k_acc(pred_labels_topk[i: i + pair_num], real_labels[i: i + pair_num], 5)
        hit_10_num += top_k_acc(pred_labels_topk[i: i + pair_num], real_labels[i: i + pair_num], 10)
        hit_1_ratio += top_k_hit_ratio(pred_labels_topk[i: i + pair_num], real_labels[i: i + pair_num], 1)
        hit_5_ratio += top_k_hit_ratio(pred_labels_topk[i: i + pair_num], real_labels[i: i + pair_num], 5)
        hit_10_ratio += top_k_hit_ratio(pred_labels_topk[i: i + pair_num], real_labels[i: i + pair_num], 10)
        # print(roc_auc_list[-1], prc_auc_list[-1])
        i += pair_num
    return (roc_auc_list, prc_auc_list,
            hit_1_num / query_num, hit_5_num / query_num, hit_10_num / query_num,
            hit_1_ratio / query_num, hit_5_ratio / query_num, hit_10_ratio / query_num,
            )


def top_k_acc(pred_labels, real_labels, k):
    idx_list = np.argsort(pred_labels)
    for i in range(k):
        # print(pred_labels[idx_list[i]], real_labels[idx_list[i]], tms[idx_list[i]])
        if real_labels[idx_list[i]] == 1:
            return 1
    return 0


def top_k_hit_ratio(pred_labels, real_labels, k):
    idx_list = np.argsort(pred_labels)
    hit_num = 0
    for i in range(k):
        # print(pred_labels[idx_list[i]], real_labels[idx_list[i]], tms[idx_list[i]])
        if real_labels[idx_list[i]] == 1:
            hit_num += 1
    return hit_num / min(k, sum(real_labels))


def cal_prc_curve(pred_labels, real_labels):
    pairs = zip(pred_labels, real_labels)
    s_pairs = sorted(pairs, key=lambda x: x[0])
    pre_list, rec_list = [], []
    for pair in s_pairs:
        th = pair[0]
        tmp = []
        for pred in pred_labels:
            if pred < th:
                tmp.append(1)
            else:
                tmp.append(0)
        cm = confusion_matrix(real_labels, tmp, labels=[1, 0])
        if cm[0][0] == 0:
            pre = 0
            rec = 0
        else:
            pre = cm[0][0] / (cm[0][0] + cm[1][0])
            rec = cm[0][0] / (cm[0][0] + cm[0][1])
        if not pre_list:
            pre_list.append(pre)
            rec_list.append(rec)
        else:
            if pre != pre_list[-1] and rec != rec_list[-1]:
                pre_list.append(pre)
                rec_list.append(rec)
            elif rec == rec_list[-1] and pre > pre_list[-1]:
                pre_list[-1] = pre
    print(rec_list)
    print(pre_list)
    return rec_list, pre_list


def cal_distance(x1, x2, mode, n_des_max=1000, len_1=None, len_2=None):
    sim_mat = torch.zeros((x1.shape[0], x2.shape[0]))
    if mode == 'euc':
        x1 = x1.unsqueeze(1)
    for i in range(0, x1.shape[0], n_des_max):
        if i + n_des_max < x1.shape[0]:
            if mode == 'euc':
                sim_mat[i:i + n_des_max, :] = euclidean_distance(x1[i:i + n_des_max, :, :], x2)
            elif mode == 'lsc':
                sim_mat[i:i + n_des_max, :] = len_scaling_cos_dist(x1[i:i + n_des_max, :, :], x2, len_1, len_2)
            elif mode == 'cos':
                sim_mat[i:i + n_des_max, :] = cos_dist(x1[i:i + n_des_max, :], x2)
        else:
            if mode == 'euc':
                sim_mat[i:, :] = euclidean_distance(x1[i:, :, :], x2)
            elif mode == 'lsc':
                sim_mat[i:, :] = len_scaling_cos_dist(x1[i:, :, :], x2, len_1, len_2)
            elif mode == 'cos':
                sim_mat[i:, :] = cos_dist(x1[i:, :], x2)
    return sim_mat


def len_scaling_cos_dist(x1: torch.Tensor, x2: torch.Tensor, len_1: list, len_2: list) -> torch.Tensor:
    """
    :param x1: descriptors of query structures
    :param x2: descriptors of compared structures
    :param len_1: length list of query structures
    :param len_2: length list of compared structures
    :return: distance matrix
    """
    max_num = 1000
    dist_mat = torch.zeros((x1.shape[0], x2.shape[0]))
    len_1_t, len_2_t = torch.tensor(len_1, dtype=torch.float), torch.tensor(len_2, dtype=torch.float)
    max_len = len_2_t.max()
    for i in range(0, x1.shape[0], max_num):
        if i + max_num < x1.shape[0]:
            dist_mat[i:i + max_num, :] = cos_dist(x1[i:i + max_num, :], x2)
        else:
            dist_mat[i:, :] = cos_dist(x1[i:, :], x2)
    len_scaling_mat = 1 + torch.clamp((- len_1_t.unsqueeze(1) + len_2_t.unsqueeze(0)) / max_len, min=0)
    return dist_mat / len_scaling_mat


def cos_dist(x1, x2):
    x1_norm = torch.norm(x1, dim=1).unsqueeze(1)
    x2_norm = torch.norm(x2, dim=1).unsqueeze(0)
    norm_mat = torch.clamp_min(torch.matmul(x1_norm, x2_norm), 1e-6)
    product_mat = x1.matmul(x2.transpose(1, 0))
    return 1 - product_mat / norm_mat


def euclidean_distance(x1, x2):
    return torch.sqrt(torch.sum(torch.pow(x1 - x2, 2), 2))


def time_stat(func):

    def wrapper(*args, **kwargs):
        begin_time = time()
        res = func(*args, **kwargs)
        end_time = time()
        exec_time = (end_time - begin_time) / 60
        if torch.cuda.device_count() == 0 or (torch.cuda.device_count() >= 1 and td.get_rank() == 0):
            logging.info('Time for {}: {:.2F} min\n'.format(func.__name__, exec_time))
        return res

    return wrapper


def log_config(log_filaname):
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_filaname, mode='a')
    formatter = logging.Formatter(LOG_FORMAT)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def write_log(logger, output_info):
    if torch.cuda.device_count() == 0 or (torch.cuda.device_count() >= 1 and td.get_rank() == 0):
        logger.info(output_info)


# 16.6 min for single process; 17 min for two processes; 21.68 min for ten processes
def sort_samples_moco_offline(pos_ratio, fold_k, n_samples, bz, qz, rand_seed):
    begin_time = time()
    print(rand_seed)

    tr_pair_dict_path = "./dataset/scope207/pair_list_for_train/tr_pair_{}.pkl".format(fold_k)
    with open(tr_pair_dict_path, 'rb') as ppdf:
        pair_data = pickle.load(ppdf)
    tms_mat, sorted_pdb_fn_dict = pair_data['tms_mat'], pair_data['sorted_pdb']
    n_tr_prots = pair_data['n_tr_prots']

    result_id_list = []
    n_bz = n_samples // bz
    random.seed(rand_seed)
    keys_queue = []
    queue_len = qz // bz
    pdb_fn_list = list(tms_mat.keys())

    for k in range(n_bz):
        new_key_pool, one_batch_id_list = [], []
        print(rand_seed, k)
        for i in range(bz):
            while True:
                chosen_pdb_fn1 = random.choice(pdb_fn_list)
                chosen_pdb_fn2 = random.choice(sorted_pdb_fn_dict[chosen_pdb_fn1][:int(n_tr_prots * pos_ratio)])
                find_flag = True
                for keys_pool in keys_queue:
                    for key_pdb in keys_pool:
                        if chosen_pdb_fn1 == key_pdb or tms_mat[chosen_pdb_fn1][key_pdb] > tms_mat[chosen_pdb_fn1][chosen_pdb_fn2]:
                            find_flag = False
                            break
                    if not find_flag:
                        break
                if find_flag:
                    break
            one_batch_id_list.append((chosen_pdb_fn1, chosen_pdb_fn2))
            new_key_pool.append(chosen_pdb_fn2)
        result_id_list.extend(one_batch_id_list)
        keys_queue.append(new_key_pool)
        if len(keys_queue) > queue_len:
            keys_queue.pop(0)

    print(len(result_id_list) / bz)
    end_time = time()
    sort_time = (end_time - begin_time) / 60
    print('time for sorting samples: {:.2F} min\n'.format(sort_time))

    return result_id_list


def get_training_pair_list(pos_ratio, fold_k, n_epoch, batch_size, queue_size):

    training_pair_list_path = "./dataset/scope207/pair_list_for_train/tr_pair_list_{}_P{}_B{}_Q{}".format(
        fold_k, int(pos_ratio * 10), batch_size, queue_size)
    if os.path.exists(training_pair_list_path):
        with open(training_pair_list_path, 'rb') as tplf:
            result = pickle.load(tplf)
    else:
        result = []

    training_pair_lists = []
    p = Pool(6)
    start_epoch = len(result)
    for i in range(start_epoch, n_epoch):
        training_pair_lists.append(
            p.apply_async(sort_samples_moco_offline, (pos_ratio, fold_k, 130000, batch_size, queue_size, i))
        )
    p.close()
    p.join()

    for res in training_pair_lists:
        result.append(res.get())

    print(len(result))
    print(len(result[0]))

    with open(training_pair_list_path, 'wb') as tplf:
        pickle.dump(result, tplf)


class FeatureExtractor(nn.Module):
    def __init__(self, model, layers):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}
        # for layer in self.model.named_modules():
        #     print(layer)
        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str):
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: torch.Tensor, test: bool):
        _ = self.model(x, test)
        return self._features


if __name__ == '__main__':
    from multiprocessing import Pool
    """
    for i in range(1, 2):
        get_training_pair_list(0.7, i, 30, 64, 1024)
        get_training_pair_list(0.7, i, 60, 64, 1024)
        get_training_pair_list(0.7, i, 90, 64, 1024)
        get_training_pair_list(0.7, i, 120, 64, 1024)
    """
