from src.trainers.base import BaseTrainer
from src.models.model import choose_model
from src.models.worker import LrAdjustWorker,LrdWorker
from src.optimizers.gd import GD
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from random import sample
import pandas as pd
from numpy.random import uniform
from collections import defaultdict
from src.utils.worker_utils import read_data
import random
import copy
import math
import torch.optim as optim

criterion = torch.nn.CrossEntropyLoss()


class FedAvg96Trainer(BaseTrainer):
    """
    uniform sample changed!!
    """
    def __init__(self, options, dataset):
        self.model = choose_model(options)
        self.move_model_to_gpu(self.model, options)

        if options['optim'] == 'GD':
            self.optimizer = GD(self.model.parameters(), lr=options['lr'], weight_decay=options['wd'])
        elif options['optim'] == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=options['lr'], weight_decay=1e-4)
        elif options['optim'] == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=options['lr'], momentum=0.9, weight_decay=5e-4)
        elif options['optim'] == 'AdamW':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-2)
        self.num_epoch = options['num_epoch']
        worker = LrdWorker(self.model, self.optimizer, options)

        super(FedAvg96Trainer, self).__init__(options, dataset, worker=worker)

    def train(self, options):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))
        Acc = []
        self.flag = -1
        # Fetch latest flat model parameter
        self.latest_model = self.worker.get_flat_model_params().detach()
        flag_list = []
        index = []
        pice = []

        index = [param.numel() for name, param in self.model.named_parameters() if param.requires_grad]
        pice = [index[i*2] + index[i*2+1] for i in range(len(index) // 2)]
        if options['model'] == 'medcnn':
            result = []
            i = 0
            while i < len(pice):
                if i > 0 and pice[i] < 200:
                    result[-1] += pice[i]  
                else:
                    result.append(pice[i])
                i += 1
            pice = result
        pice = [sum(pice[:i+1]) for i in range(len(pice))]
        pice.insert(0,0)
        if options['model'] == 'medcnn':
            pice = [0, 12240, 49296, 86352, 348752, 351065]
        elif options['model'] == 'cifar':
            pice = [0, 49296, 86352, 496208, 498778]

        flag = True
        for round_i in range(self.num_round):
            flag_list.append(self.flag)
            print(flag_list)
            Acc.append(self.test_latest_model_on_evaldata(round_i))
            print(Acc)
            selected_clients = self.select_clients(seed=round_i)

            # print(self.metrics)
            # exit()
            client_epoch = self.adjust_epoch_gai(selected_clients, round_i,Acc,options)
            selected_clients_index = []
            for i in selected_clients:
                selected_clients_index.append(i.cid)
            solns, stats = self.local_train(round_i, selected_clients,client_epoch)
            self.update_local_gradient(pice, self.latest_model, solns, selected_clients)
            self.metrics.extend_commu_stats(round_i, stats)
            latest_model = self.aggregate(solns, round_i, pice)
            if options['model'] == 'cnn':    
                if round_i == 99:
                    print("last parament count:",self.parament_count)
                    print("$$$$$$$$$$$$:::",self.parament_count*4*options['clients_per_round']/100/1048576,"MB")
            else:
                if round_i == 199:
                    print("last parament count:",self.parament_count)
                    print("$$$$$$$$$$$$:::",self.parament_count*4*options['clients_per_round']/100/1048576,"MB")
            self.update_global_gradient(latest_model)
            if round_i >= 100 and round_i % 5 == 0:
                self.flag = self.parament_select(pice,selected_clients_index)
            self.latest_model = latest_model
            for param_group in self.optimizer.param_groups:
                print("Current learning rate:", param_group['lr'])
            if options['optim'] == 'GD' and options['model'] == 'cnn':
                self.optimizer.inverse_prop_decay_learning_rate(round_i)
            elif options['optim'] == 'GD' and (options['model'] == 'medcnn' or options['model'] == 'cif2ar' ) :
                self.optimizer.inverse_prop_decay_learning_rate_med(round_i)
            elif options['optim'] == 'GD' and options['model'] == 'cifar':
                self.optimizer.inverse_prop_decay_learning_rate_cifar(round_i)    
            for c in selected_clients:
                self.repeat_time[c.cid] += 1

            for c in selected_clients:
                self.repeat_time[c.cid] += 1



        # Test final model on train cifar-10-batches-py
        self.test_latest_model_on_traindata(self.num_round)
        self.test_latest_model_on_evaldata(self.num_round)
        print(self.repeat_time)
        print(self.sim_var)
        print(self.accs)
        print("avg scales", np.average(self.scales))
        print("client label in each round", self.lables)
        # Save tracked information
        self.metrics.write()
    def parament_select(self, pice,selected_clients_index):
        parament_flag = []

        for i in range(len(pice)-1):
            avg = []
            for j in selected_clients_index:
                avg.append(self.get_cos_similar(self.raw_gradients[j][pice[i]:pice[i+1]],self.raw_global_update[pice[i]:pice[i+1]]))
            parament_flag.append(sum(avg)/len(avg))
        print(parament_flag,"zzzzzzzz")
        return parament_flag.index(min(parament_flag))
    
    

    def local_train(self, round_i, selected_clients, client_epochs, **kwargs):
        solns = []  # Buffer for receiving client solutions
        stats = []  # Buffer for receiving client communication costs
        for i, c in enumerate(selected_clients, start=1):

            # Communicate the latest model
            c.set_flat_model_params(self.latest_model)
            # Solve minimization locally
            m = len(c.train_data) / self.all_train_data_num * 100
            soln, stat = c.local_train(self.latest_model,num_epoch=client_epochs[i - 1] if client_epochs is not None else None, multiplier=m)

            if self.print_result:
                print("Round: {:>2d} | CID: {: >3d} ({:>2d}/{:>2d})| "
                      "Param: norm {:>.4f} ({:>.4f}->{:>.4f})| "
                      "Loss {:>.4f} | Acc {:>5.2f}% | Time: {:>.2f}s".format(
                    round_i, c.cid, i, self.clients_per_round,
                    stat['norm'], stat['min'], stat['max'],
                    stat['loss'], stat['acc'] * 100, stat['time']))

            # Add solutions and stats
            solns.append(soln)
            stats.append(stat)
    
        return solns, stats
    def alpha(self, accuracy_prev):
        accuracy_min = 0.05
        accuracy_max = 0.93
        # 定义 α 的范围
        α_min = 0.07
        α_max = 0.66

        # 定义指数函数的敏感度常数
        k = 9.0
        # 假设上一轮 server 端模型的准确率为 accuracy_prev
        # 计算归一化准确率
        normalized_accuracy = (accuracy_prev - accuracy_min) / (accuracy_max - accuracy_min)

        # 使用指数函数调整 α
        α_exp = math.exp(k * normalized_accuracy)

        # 计算指数函数计算出的 α 的最小值和最大值
        α_min_exp = math.exp(k * 0)  # 当 normalized_accuracy = 0 时
        α_max_exp = math.exp(k * 1)  # 当 normalized_accuracy = 1 时

        # 归一化处理
        α = α_min + (α_max - α_min) * (α_exp - α_min_exp) / (α_max_exp - α_min_exp)
        return α