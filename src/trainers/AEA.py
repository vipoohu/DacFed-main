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


class FedAvg9Trainer(BaseTrainer):
    """
    uniform sample changed!!
    """
    def __init__(self, options, dataset):
        self.model = choose_model(options)
        self.move_model_to_gpu(self.model, options)

        if options['optim'] == 'GD':
            print("GGGGGGGGGGGGGD")
            self.optimizer = GD(self.model.parameters(), lr=options['lr'], weight_decay=options['wd'])
        elif options['optim'] == 'Adam':
            print("Adam!!!!!!!!!!!!")
            self.optimizer = optim.Adam(self.model.parameters(), lr=options['lr'], weight_decay=1e-4)
        elif options['optim'] == 'SGD':
            print("SGD!!!!!!!!!!!!!!")
            self.optimizer = optim.SGD(self.model.parameters(), lr=options['lr'], momentum=0.9, weight_decay=5e-4)
        elif options['optim'] == 'AdamW':
            print("AdamWWWW!!!!!!!!!!!!")
            self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-2)
        self.num_epoch = options['num_epoch']
        # worker = LrAdjustWorker(model, self.optimizer, options)
        worker = LrdWorker(self.model, self.optimizer, options)

        super(FedAvg9Trainer, self).__init__(options, dataset, worker=worker)

    def train(self, options):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))
        Acc = []
        # Fetch latest flat model parameter
        self.latest_model = self.worker.get_flat_model_params().detach()
        
        flag = True
        for round_i in range(self.num_round):
            
            Acc.append(self.test_latest_model_on_evaldata(round_i))
            print(Acc)
            selected_clients = self.select_clients(seed=round_i)
            client_epoch = self.adjust_epoch_gai(selected_clients, round_i,Acc,options)
            pice = []

            solns, stats = self.local_train(round_i, selected_clients,client_epoch)
            self.update_local_gradient(pice, self.latest_model, solns, selected_clients)
            # Track communication cost
            self.metrics.extend_commu_stats(round_i, stats)
            
            # Update latest model

            if round_i <= 600 :
                latest_model = self.aggregate(solns)
            else:
                latest_model = self.aggregate_lambda(solns,stats,Acc,round_i)
            self.update_global_gradient(latest_model)
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
        # Save tracked information
        self.metrics.write()

    def aggregate_lambda(self, solns, stats, Acc, round_i):
        select_num = 10
        averaged_solution = torch.zeros_like(self.latest_model)
        accum_sample_num = 0
        temp = []
        for i in stats:
            temp.append([i['id'], i['loss']])
        sorted_temp = sorted(temp, key = lambda x: x[1], reverse = True)
        model_parament = []
        for i in temp:
            model_parament.append([i[0], self.get_cos_similar(self.gradients[i[0]], self.global_update)])
        sorted_model_parament = sorted(model_parament, key = lambda x: x[1], reverse = True)
        set_loss = set()
        set_gradient = set()
        for i in range(select_num):
            set_loss.add(sorted_temp[i][0])
            set_gradient.add(sorted_model_parament[i][0])
        intersection_set = set_loss.intersection(set_gradient)
        select_aggregate = list(intersection_set)
        select_aggregate_origin = copy.deepcopy(select_aggregate)
        if len(select_aggregate) < select_num:
            for i in sorted_temp:
                if i[0] not in select_aggregate:
                    select_aggregate.append(i[0])
                if len(select_aggregate) == select_num:
                    break
        mmax = 0
        def find_loss(id, temp):
            for i in temp:
                if id == i[0]:
                    return i[1]
                else:
                    continue
        for num_sample, id, local_solution in solns:
            if id in select_aggregate:
                accum_sample_num +=(find_loss(id, temp)*3 + (1- find_loss(id, model_parament)))
                averaged_solution += (find_loss(id, temp)*3 + (1- find_loss(id, model_parament))) * (local_solution - self.latest_model)

        averaged_solution /= accum_sample_num
        averaged_solution += self.latest_model
        α = self.alpha(Acc[round_i])
        α = 0
        averaged_solution = (1 - α) * averaged_solution + α * self.latest_model
        return averaged_solution.detach()
    def aggregate(self, solns, client_epoch=None, **kwargs):
        averaged_solution = torch.zeros_like(self.latest_model)
        # averaged_solution = np.zeros(self.latest_model.shape)
        assert self.simple_average

        for i, (num_sample,id, local_solution) in enumerate(solns):
            averaged_solution += local_solution
        averaged_solution /= len(solns)

        return averaged_solution.detach()

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
        α_min = 0.07
        α_max = 0.66

        k = 9.0
        normalized_accuracy = (accuracy_prev - accuracy_min) / (accuracy_max - accuracy_min)

        α_exp = math.exp(k * normalized_accuracy)

        α_min_exp = math.exp(k * 0)  # 当 normalized_accuracy = 0 时
        α_max_exp = math.exp(k * 1)  # 当 normalized_accuracy = 1 时
        α = α_min + (α_max - α_min) * (α_exp - α_min_exp) / (α_max_exp - α_min_exp)
        return α