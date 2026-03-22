import os
import argparse

import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as td
from distributed_sampler import DistributedSampler
# import matplotlib.pyplot as plt

from cnn import CNN
from model import Encoder
from moco import MoCo
from data import SCOPeTrain, SCOPeTest, NewRelease
from util import eval_metric, cal_distance, time_stat, write_log, log_config, len_scaling_cos_dist, FeatureExtractor


def collate_fn_padd(batch):
    data, len_data = [], []
    for i in range(2):
        tmp = [item[i] for item in batch]
        df = nn.utils.rnn.pad_sequence(tmp, batch_first=True)
        len_list = [dv.shape[0] for dv in tmp]
        data.append(df)
        len_data.append(torch.tensor(len_list, dtype=torch.int))

    # adjacency matrix
    adj_mat = []
    for i in range(2, 4):
        tmp = [item[i] for item in batch]
        max_len = torch.max(len_data[i - 2])
        d_list = []
        for am in tmp:
            am_len = am.shape[0]
            if am_len < max_len:
                d_list.append(torch.cat(
                    (torch.from_numpy(am).float(), torch.zeros((am_len, max_len - am_len))), dim=1))
            else:
                d_list.append(torch.from_numpy(am).float())
        adj_mat.append(nn.utils.rnn.pad_sequence(d_list, batch_first=True))

    return [data[0], data[1], len_data[0], len_data[1], adj_mat[0], adj_mat[1]]


def collate_fn_padd_cnn(batch):
    len_data = []
    for i in range(2):
        tmp = [item[i] for item in batch]
        len_list = [dv.shape[0] for dv in tmp]
        len_data.append(torch.tensor(len_list, dtype=torch.int))

    # inverse distance matrix
    adj_mat = []
    for i in range(2, 4):
        tmp = [item[i] for item in batch]
        max_len = torch.max(len_data[i - 2])
        d_list = []
        for am in tmp:
            am_len = am.shape[0]
            if am_len < max_len:
                d_list.append(torch.cat(
                    (torch.from_numpy(am).float(), torch.zeros((am_len, max_len - am_len))), dim=1))
            else:
                d_list.append(torch.from_numpy(am).float())
        adj_mat.append(nn.utils.rnn.pad_sequence(d_list, batch_first=True, padding_value=0))

    return [torch.zeros((1,)), torch.zeros((1,)), len_data[0], len_data[1], adj_mat[0], adj_mat[1]]


class TrainTest:
    @time_stat
    def __init__(self, is_training, is_retraining, test_set_name, fold_k, train_batch_size, train_queue_size,
                 lr, len_scaling, dyna_par, use_cnn, model_path, logger):
        self.n_ref_points = 31
        self.logger = logger
        self.len_scaling = len_scaling
        self.use_cnn = use_cnn
        self.fold_k = fold_k
        scope_fp = 'dataset/scope207/features/scope_fn_xyz_a.npz'

        if torch.cuda.device_count() >= 1:
            td.init_process_group(backend="nccl")
            self.local_rank = td.get_rank()
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            write_log(logger, 'card: {}'.format(os.environ["CUDA_VISIBLE_DEVICES"]))

        write_log(logger, "Model path: {}".format(model_path))
        if is_training:
            write_log(logger, "Batch size: {} Queue size: {}".format(train_batch_size, train_queue_size))
            write_log(logger, "Initial learning rate: {}".format(lr))
            write_log(logger, "Using length-scaling: {} Using dynamic partition: {}\n".format(
                len_scaling, dyna_par))

            self.train_data = SCOPeTrain(train_batch_size, train_queue_size, (scope_fp,), self.n_ref_points,
                                         use_cnn, logger, fold_k, dyna_par)
            self.sampler = DistributedSampler(self.train_data, shuffle=False)
            if use_cnn:
                self.train_dataloader = DataLoader(
                    dataset=self.train_data, batch_size=train_batch_size // td.get_world_size(),
                    collate_fn=collate_fn_padd_cnn, shuffle=False, sampler=self.sampler, pin_memory=True)
            else:
                self.train_dataloader = DataLoader(
                    dataset=self.train_data, batch_size=train_batch_size // td.get_world_size(),
                    collate_fn=collate_fn_padd, shuffle=False, sampler=self.sampler, pin_memory=True)
        if use_cnn:
            self.net = MoCo(self.n_ref_points, CNN, 400, train_queue_size, 0.999, 0.07, use_cnn=True)
        else:
            self.net = MoCo(self.n_ref_points, Encoder, 400, train_queue_size, 0.999, 0.07)
        if torch.cuda.device_count() >= 1:
            self.net.to(self.device)
        write_log(self.logger, "Initial learning rate: {}".format(lr))
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', 0.1, 10, verbose=True)

        if torch.cuda.device_count() >= 1:
            write_log(self.logger, "Let's use {} GPUs!".format(torch.cuda.device_count()))
            self.net = nn.parallel.DistributedDataParallel(self.net, device_ids=[self.local_rank],
                                                           output_device=self.local_rank,
                                                           find_unused_parameters=True)
        if is_retraining:
            td.barrier()
            map_location = {'cuda:%d' % 0: 'cuda:%d' % self.local_rank}
            self.net.load_state_dict(torch.load(model_path, map_location=map_location))

        self.TEST_BATCH_SIZE = 32
        # load test set
        if test_set_name == 'scope':
            self.test_data = SCOPeTest(-1, (scope_fp,), self.n_ref_points, use_cnn, self.logger, fold_k, 'Train')
        elif test_set_name == 'new':
            newr_fp = 'dataset/new_release/features/new_fn_xyz_a.npz'
            self.test_data = NewRelease(-1, (newr_fp, scope_fp), self.n_ref_points, use_cnn, self.logger, 'Train')
        else:
            raise RuntimeError('No dataset {}'.format(test_set_name))

        # define shared var
        self.model_path = model_path
        self.n_fea_types = 6
        self.tr_descriptors = None
        self.iter_num = 0

    def training(self, n_epoch, epoch_base):
        max_prc = 0 if epoch_base == 0 else 0.55
        for epoch in range(epoch_base, epoch_base + n_epoch):
            write_log(self.logger, "Training...")
            train_loss = self.train_one_epoch(epoch)

            # validating
            tmp, top_prc = self.testing()
            if top_prc > max_prc:
                max_prc = top_prc
                if self.local_rank == 0:
                    torch.save(self.net.state_dict(), self.model_path)
                    write_log(self.logger, "Saved! current loss: {},  PRCAUC: {}".format(train_loss, max_prc))

            prev_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.scheduler.step(top_prc)
            cur_lr = self.optimizer.state_dict()['param_groups'][0]['lr']

            if prev_lr != cur_lr:
                td.barrier()
                map_location = {'cuda:%d' % 0: 'cuda:%d' % self.local_rank}
                self.net.load_state_dict(torch.load(self.model_path, map_location=map_location))
                write_log(self.logger, "Load model!")

            if cur_lr < 5e-5:
                return

    @time_stat
    def train_one_epoch(self, epoch):
        self.net.train()
        training_loss_sum = 0
        loss_history = torch.tensor([0, 0], dtype=torch.float16).to(self.device)  # loss_sum
        n_iter_per_epoch = 500
        for i, data in enumerate(self.train_dataloader):
            for j in range(self.n_fea_types):
                data[j] = data[j].to(self.device)
            predictions, labels = self.net(data[:self.n_fea_types], False)
            self.optimizer.zero_grad()
            loss = self.criterion(predictions, labels)
            loss.backward()
            training_loss_sum += float(loss.data)
            self.optimizer.step()

            if i % n_iter_per_epoch == n_iter_per_epoch - 1:
                self.iter_num += n_iter_per_epoch
                write_log(self.logger, "Rank: {}, Epoch:{}, iteration: {}, Current loss {}".format(
                    self.local_rank, epoch, i + 1, training_loss_sum / n_iter_per_epoch))
                loss_history[0] += training_loss_sum / float(n_iter_per_epoch)
                loss_history[1] += 1.0
                training_loss_sum = 0
        write_log(self.logger, "Current learning rate: {}".format(
            self.optimizer.state_dict()['param_groups'][0]['lr']))
        td.barrier()
        td.all_reduce(loss_history, td.ReduceOp.SUM)
        mean_loss = loss_history[0] / loss_history[1]
        write_log(self.logger, "Epoch:{}, total loss {}".format(epoch, mean_loss))

        return mean_loss

    @torch.no_grad()
    @time_stat
    def testing(self, predict_only=False):
        write_log(self.logger, "eval...")

        # load model if prediction only
        if predict_only:
            self.load_model()
        self.net.eval()
        # get training descriptors
        self.tr_descriptors, self.tr_len_list = self.get_descriptors('Train')
        write_log(self.logger, self.tr_descriptors.shape)
        # get testing descriptors
        te_descriptors, te_len_list = self.get_descriptors('Test')
        write_log(self.logger, te_descriptors.shape)
        # get labels
        label_list = self.test_data.get_labels()
        if self.len_scaling:
            dist_mat = len_scaling_cos_dist(te_descriptors, self.tr_descriptors, te_len_list, self.tr_len_list)
        else:
            dist_mat = cal_distance(te_descriptors, self.tr_descriptors, 'cos')
        dist_list = dist_mat.reshape(-1).tolist()
        # eval & show result
        pair_num = self.test_data.pair_num_list[0]
        write_log(self.logger, "pair num: {}".format(pair_num))
        avg_prc_auc = self.print_pred_info(dist_list, label_list, pair_num)

        return te_descriptors, avg_prc_auc

    def save_residue_level_descriptors(self):
        self.load_model()
        self.net.eval()
        # get training descriptors
        self.test_data.select_set("Train")
        self.test_dataloader = DataLoader(
            dataset=self.test_data, num_workers=4, batch_size=1,
            collate_fn=collate_fn_padd, shuffle=False)
        tr_des = self.get_residue_level_descriptors(self.test_dataloader)
        # get testing descriptors
        self.test_data.select_set("Test")
        self.test_dataloader = DataLoader(
            dataset=self.test_data, num_workers=4, batch_size=1,
            collate_fn=collate_fn_padd, shuffle=False)
        te_des = self.get_residue_level_descriptors(self.test_dataloader)

        te_des_dict, tr_des_dict = {}, {}
        tr_id_list = [t[0] for t in self.test_data.train_samples]
        print(len(tr_id_list))
        te_id_list = [t[0] for t in self.test_data.test_samples]
        print(len(te_id_list))
        for i, te_id in enumerate(te_id_list):
            te_des_dict[te_id] = te_des[i]
        for i, tr_id in enumerate(tr_id_list):
            tr_des_dict[tr_id] = tr_des[i]

        des_dict = {"test": te_des_dict, "train": tr_des_dict}
        with open("descriptors/scope_residue_level_fold{}.pkl".format(self.fold_k), 'wb') as df:
            pickle.dump(des_dict, df)

    def load_model(self):
        if torch.cuda.device_count() >= 1:
            td.barrier()
            map_location = {'cuda:%d' % 0: 'cuda:%d' % self.local_rank}
            self.net.load_state_dict(torch.load(self.model_path, map_location=map_location))
        else:
            state_dict_cpu = {k.replace("module.", ""): v for k, v in torch.load(
                self.model_path, map_location='cpu').items()}
            self.net.load_state_dict(state_dict_cpu)

    @time_stat
    def save_scope_descriptors(self):
        te_des_dict, tr_des_dict = {}, {}
        te_des, _ = self.testing(True)
        tr_id_list = [t[0] for t in self.test_data.train_samples]
        print(len(tr_id_list))
        te_id_list = [t[0] for t in self.test_data.test_samples]
        print(len(te_id_list))
        for i, te_id in enumerate(te_id_list):
            te_des_dict[te_id] = torch.cat((te_des[i, :], torch.tensor(
                self.test_data.node_fea_dict[te_id].shape[0], dtype=torch.float).reshape((1,))))
        for j, tr_id in enumerate(tr_id_list):
            tr_des_dict[tr_id] = torch.cat((self.tr_descriptors[j, :], torch.tensor(
                self.test_data.node_fea_dict[tr_id].shape[0], dtype=torch.float).reshape((1,))))
        des_dict = {"test": te_des_dict, "train": tr_des_dict}
        model_name = os.path.basename(self.model_path).split('.')[0]
        with open("descriptors/scope_{}.pkl".format(model_name), 'wb') as df:
            pickle.dump(des_dict, df)

    def get_residue_level_descriptors(self, pred_dataloader):
        pred_descriptors = []
        for data in pred_dataloader:
            with torch.no_grad():
                layer_id = "encoder_q.gcrb_2"
                if torch.cuda.device_count() >= 1:
                    for j in range(self.n_fea_types):
                        data[j] = data[j].to(self.device)
                    layer_id = "module.encoder_q.gcrb_2"
                fe = FeatureExtractor(self.net, (layer_id,))
                residue_level_descriptors = fe(data[:self.n_fea_types], True)[layer_id].squeeze()
                pred_descriptors.append(residue_level_descriptors.cpu())
        return pred_descriptors

    def get_descriptors(self, mode):
        self.test_data.select_set(mode)
        if self.use_cnn:
            self.test_dataloader = DataLoader(
                dataset=self.test_data, num_workers=4, batch_size=self.TEST_BATCH_SIZE,
                collate_fn=collate_fn_padd_cnn, shuffle=False)
        else:
            self.test_dataloader = DataLoader(
                dataset=self.test_data, num_workers=4, batch_size=self.TEST_BATCH_SIZE,
                collate_fn=collate_fn_padd, shuffle=False)
        return self.predict(self.test_dataloader)

    def predict(self, pred_dataloader):
        pred_descriptors = []
        len_list = []
        for data in pred_dataloader:
            with torch.no_grad():
                if torch.cuda.device_count() >= 1:
                    for j in range(self.n_fea_types):
                        data[j] = data[j].to(self.device)
                predictions = self.net(data[:self.n_fea_types], True)
                pred_descriptors.append(predictions.cpu())
                len_list.extend(data[2])
        return torch.cat(pred_descriptors, 0), len_list

    def print_pred_info(self, dist_list, label_list, pair_num):
        metrics = eval_metric(dist_list, label_list, pair_num, verse=True)
        write_log(self.logger, 'No. of testing samples: {}'.format(len(metrics[0])))
        avg_roc_auc = sum(metrics[0]) / len(metrics[0])
        avg_prc_auc = sum(metrics[1]) / len(metrics[1])
        write_log(self.logger, 'average ROCAUC: {}\t average PRCAUC: {}\n'
                               'top 1 accuracy: {}\t top 5 accuracy: {}\t top 10 accuracy: {}\n'
                               'top 1 hit ratio: {}\t top 5 hit ratio: {}\t top 10 hit ratio: {}\n'.format(
                                avg_roc_auc, avg_prc_auc, metrics[2], metrics[3], metrics[4],
                                metrics[5], metrics[6], metrics[7]))
        return avg_prc_auc


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('-m', '--model_name', type=str, help='name of the model to be stored.')
    parser.add_argument('-t', '--train', help='train or predict the model', action='store_true')
    parser.add_argument('-r', '--retrain', help='retrain the model or not', action='store_true')
    parser.add_argument('-s', '--epoch_base', type=int, default=0,
                        help='The no. of last epoch before retraining (only used for retraining)')
    parser.add_argument('-d', '--dataset_name', type=str, help='name of dataset')
    parser.add_argument('-f', '--fold', type=int, help='current fold of cross-validation')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='batch size (only used for training)')
    parser.add_argument('-q', '--queue_size', type=int, default=1024,
                        help='queue size must be multiple of batch size (only used for training)')
    parser.add_argument('-e', '--epoch_num', type=int, default=120,
                        help='the number of epoch (only used for training)')
    parser.add_argument('-l', '--learning_rate', type=float, default=1e-1,
                        help='learning rate (only used for training)')
    parser.add_argument('--save_descriptor', type=str, default="",
                        help="save global or local descriptors of SCOPe of the current fold in './descriptors'")
    parser.add_argument('--length_scaling', help='apply length-scaling', action='store_true')
    parser.add_argument('--dynamic_partition', help='apply dynamic partition', action='store_true')
    parser.add_argument("--cnn", help='use CNN as encoders', action='store_true')
    return parser.parse_args()


def main():
    args = parse_argument()
    logger = log_config('{}.log'.format(args.model_name))
    model_path = 'saved_model/{}.pkl'.format(args.model_name)
    assert args.queue_size % args.batch_size == 0
    if args.retrain and args.epoch_base == 0:
        raise RuntimeError("Epoch base needs to be set when retraining!")
    if not args.retrain and args.epoch_base != 0:
        raise RuntimeError("Epoch base shouldn't be set!")

    tt = TrainTest(args.train, args.retrain, args.dataset_name, args.fold, args.batch_size, args.queue_size,
                   args.learning_rate, args.length_scaling, args.dynamic_partition, args.cnn,
                   model_path, logger)
    if args.train:
        tt.training(args.epoch_num, args.epoch_base)
    elif args.save_descriptor == "global":
        tt.save_scope_descriptors()
    elif args.save_descriptor == "local":
        tt.save_residue_level_descriptors()
    else:
        tt.testing(True)


if __name__ == '__main__':
    main()
