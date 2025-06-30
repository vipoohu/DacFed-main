import pickle
import json
import numpy as np
import os
import time
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F

__all__ = ['mkdir', 'read_data', 'Metrics', "MiniDataset"]


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def read_data(train_data_dir, test_data_dir, key=None):
    """Parses cifar-10-batches-py in given train and test cifar-10-batches-py directories

    Assumes:
        1. the cifar-10-batches-py in the input directories are .json files with keys 'users' and 'user_data'
        2. the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train cifar-10-batches-py (ndarray)
        test_data: dictionary of test cifar-10-batches-py (ndarray)
    """

    clients = []
    groups = []
    train_data = {}
    test_data = {}
    print('>>> Read cifar-10-batches-py from:')

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.pkl')]
    flag = 0
    if (str(train_data_dir)=='./data/femnist/data/train'):
        flag = 1
    elif (str(train_data_dir)=='./data/medmnist/data/train'):
        flag = 2
    if key is not None:
        train_files = list(filter(lambda x: str(key) in x, train_files))

    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        print('    ', file_path)

        with open(file_path, 'rb') as inf:
            cdata = pickle.load(inf)
        if flag == 0 or 2:
            clients.extend(cdata['users'])
        elif flag == 1:
            clients.extend(cdata['users'][:500])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        if flag == 1:
            filtered_dict = {key: value for key, value in cdata['user_data'].items() if key <500}
            train_data.update(filtered_dict)
        elif flag == 0 or 2:
            train_data.update(cdata['user_data'])
    flag_cifar = 0
    for cid, v in train_data.items():
        train_data[cid] = MiniDataset(v['x'], v['y'],flag, flag_cifar)

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.pkl')]
    if key is not None:
        test_files = list(filter(lambda x: str(key) in x, test_files))

    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        print('    ', file_path)

        with open(file_path, 'rb') as inf:
            cdata = pickle.load(inf)
        if flag == 1 :
            filtered_dict = {key: value for key, value in cdata['user_data'].items() if key <500}
            test_data.update(filtered_dict)
        elif flag == 0 or 2:
            test_data.update(cdata['user_data'])


        
    flag_cifar = 1
    for cid, v in test_data.items():
        test_data[cid] = MiniDataset(v['x'], v['y'], flag,flag_cifar)
    clients = list(sorted(train_data.keys()))

    return clients, groups, train_data, test_data


class MiniDataset(Dataset):
    def __init__(self, data, labels,flag,flag_cifar):
        super(MiniDataset, self).__init__()
        
        self.data = np.array(data)
        self.labels = np.array(labels).astype("int64")

        # self.data = self.data.astype("uint8")
        # # self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # self.transform = transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Lambda(lambda x: F.pad(
        #             Variable(x.unsqueeze(0), requires_grad=False),
        #             (4, 4, 4, 4), mode='reflect').data.squeeze()),
        #         transforms.ToPILImage(),
        #         transforms.RandomCrop(32),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #     ])
        # print(self.data.ndim,"3333333333333333")
        # print(self.data.shape)
        # print(len(self.data[0]))
        # exit()

        if self.data.ndim == 4 and self.data.shape[2] == 32 and self.data.shape[3] == 3 and flag_cifar == 0:
            self.data = self.data.astype("uint8")
            self.transform = transforms.Compose(
                [
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                 ]
            )
        elif self.data.ndim == 4 and self.data.shape[2] == 28:
            self.data = self.data.astype("uint8")
            self.transform = transforms.Compose(
                [
                transforms.ToTensor(),
                transforms.Normalize((0.706, 0.506, 0.706), (0.235, 0.279, 0.216))
                 ]
            )
        elif self.data.ndim == 4 and self.data.shape[2] == 32 and self.data.shape[3] == 3 and flag_cifar == 1:
            self.data = self.data.astype("uint8")
            self.transform = transforms.Compose(
                [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                 ]
            )

        elif self.data.ndim == 4 and self.data.shape[3] == 1:
            self.transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))
                 ]
            )

        elif self.data.ndim == 3 and flag == 0:
            self.data = self.data.reshape(-1, 28, 28, 1).astype("uint8")
            self.transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))
                 ]
            )
        # elif self.data.ndim == 3 and flag == 2:

        #     self.data = self.data.reshape(-1, 28, 28, 1).astype("uint8")
        #     self.transform = transforms.Compose(
        #         [transforms.ToTensor(),
        #          transforms.Normalize(mean=[.5], std=[.5])
        #          ]
        #     )
        elif self.data.ndim == 3 and flag == 1:
            self.data = self.data.reshape(-1, 28, 28).astype("float32")
            # print(self.data)
            self.transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                 ]
            )
        else:

            self.data = self.data.astype("float32")
            self.transform = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data, target = self.data[index], self.labels[index]

        if self.data.ndim == 4 and self.data.shape[3] == 3:
            data = Image.fromarray(data)

        if self.transform is not None:
            data = self.transform(data)

        return data, target


class Metrics(object):
    def __init__(self, clients, options, name=''):
        self.options = options
        num_rounds = options['num_round'] + 1
        self.bytes_written = {c.cid: [0] * num_rounds for c in clients}
        self.client_computations = {c.cid: [0] * num_rounds for c in clients}
        self.bytes_read = {c.cid: [0] * num_rounds for c in clients}

        # Statistics in training procedure
        self.loss_on_train_data = [0] * num_rounds
        self.acc_on_train_data = [0] * num_rounds
        self.gradnorm_on_train_data = [0] * num_rounds
        self.graddiff_on_train_data = [0] * num_rounds

        # Statistics in test procedure
        self.loss_on_eval_data = [0] * num_rounds
        self.acc_on_eval_data = [0] * num_rounds

        self.result_path = mkdir(os.path.join('./result', self.options['dataset']))
        suffix = '{}_sd{}_lr{}_epoch{}_bs{}_AA{}_{}'.format(name,
                                                    options['seed'],
                                                    options['lr'],
                                                    options['num_epoch'],
                                                    options['batch_size'],
                                                    options['dynamic_E'],
                                                    options['note'])
        # suffix='{}_sd{}_lr{}_ep{}_bs{}_{}'.format(name,options['seed'],options['lr'],options['num_epoch'],
        #                                             options['batch_size'],
        #                                             'w' if options['noaverage'] else 'a')

        self.exp_name = '{}_{}_{}_{}'.format(time.strftime('%Y-%m-%dT%H-%M-%S'), options['algo'],
                                             options['model'], suffix)
        # print(options['note'])

        if options['dis']:
            suffix = options['dis']
            self.exp_name += '_{}'.format(suffix)
        train_event_folder = mkdir(os.path.join(self.result_path, self.exp_name, 'train.event'))
        eval_event_folder = mkdir(os.path.join(self.result_path, self.exp_name, 'eval.event'))
        self.train_writer = SummaryWriter(train_event_folder)
        self.eval_writer = SummaryWriter(eval_event_folder)

    def update_commu_stats(self, round_i, stats):
        cid, bytes_w, comp, bytes_r = \
            stats['id'], stats['bytes_w'], stats['comp'], stats['bytes_r']

        self.bytes_written[cid][round_i] += bytes_w
        self.client_computations[cid][round_i] += comp
        self.bytes_read[cid][round_i] += bytes_r

    def extend_commu_stats(self, round_i, stats_list):
        for stats in stats_list:
            self.update_commu_stats(round_i, stats)

    def update_train_stats(self, round_i, train_stats):
        self.loss_on_train_data[round_i] = train_stats['loss']
        self.acc_on_train_data[round_i] = train_stats['acc']
        self.gradnorm_on_train_data[round_i] = train_stats['gradnorm']
        self.graddiff_on_train_data[round_i] = train_stats['graddiff']

        self.train_writer.add_scalar('train_loss', train_stats['loss'], round_i)
        self.train_writer.add_scalar('train_acc', train_stats['acc'], round_i)
        self.train_writer.add_scalar('gradnorm', train_stats['gradnorm'], round_i)
        self.train_writer.add_scalar('graddiff', train_stats['graddiff'], round_i)

    def update_eval_stats(self, round_i, eval_stats):
        self.loss_on_eval_data[round_i] = eval_stats['loss']
        self.acc_on_eval_data[round_i] = eval_stats['acc']

        self.eval_writer.add_scalar('test_loss', eval_stats['loss'], round_i)
        self.eval_writer.add_scalar('test_acc', eval_stats['acc'], round_i)

    def write(self):
        metrics = dict()

        # String
        metrics['dataset'] = self.options['dataset']
        metrics['num_round'] = self.options['num_round']
        metrics['eval_every'] = self.options['eval_every']
        metrics['lr'] = self.options['lr']
        metrics['num_epoch'] = self.options['num_epoch']
        metrics['batch_size'] = self.options['batch_size']

        metrics['loss_on_train_data'] = self.loss_on_train_data
        metrics['acc_on_train_data'] = self.acc_on_train_data
        metrics['gradnorm_on_train_data'] = self.gradnorm_on_train_data
        metrics['graddiff_on_train_data'] = self.graddiff_on_train_data

        metrics['loss_on_eval_data'] = self.loss_on_eval_data
        metrics['acc_on_eval_data'] = self.acc_on_eval_data

        # Dict(key=cid, value=list(stats for each round))
        metrics['client_computations'] = self.client_computations
        metrics['bytes_written'] = self.bytes_written
        metrics['bytes_read'] = self.bytes_read

        metrics_dir = os.path.join(self.result_path, self.exp_name, 'metrics.json')

        with open(metrics_dir, 'w') as ouf:
            json.dump(str(metrics), ouf)
