import sys

from src.trainers.base import BaseTrainer
from src.models.model import choose_model
from src.models.worker import LrdWorker
from itertools import islice
from src.optimizers.gd import GD
import numpy as np
import torch
import math
import random
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import defaultdict
from collections import Counter

criterion = torch.nn.CrossEntropyLoss()


class FPSTrainergai96(BaseTrainer):
    """
    FPS sample
    """
    def __init__(self, options, dataset):

        self.model = choose_model(options)

        self.move_model_to_gpu(self.model, options)
        if options['optim'] == 'GD':
            self.optimizer = GD(self.model.parameters(), lr=options['lr'], weight_decay=options['wd'])
        elif options['optim'] == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=options['lr'], weight_decay=5e-4)
        elif options['optim'] == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=options['lr'], momentum=0.9, weight_decay=5e-4)
        elif options['optim'] == 'AdamW':
            optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-2)
        self.num_epoch = options['num_epoch']
        worker = LrdWorker(self.model, self.optimizer, options)
        super(FPSTrainergai96, self).__init__(options, dataset, worker=worker)

        self.last_time = [0 for _ in range(self.num_clients)]

    def train(self,options):
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
        solns = None
        # 582026
        # print(np.shape(self.gradients[0]))
        my_dict = {i: [] for i in range(0, len(self.clients) + 1)}
        for round_i in range(self.num_round):
            flag_list.append(self.flag)
            print(flag_list)
            # self.clients_per_round = int(random.gauss(10,3))
            # Test latest model on train cifar-10-batches-py
            # self.test_latest_model_on_traindata(round_i)
            Acc.append(self.test_latest_model_on_evaldata(round_i))
            # self.clients_per_round = int(random.gauss(10, 3))

            """
            get init local and global gradients
            """
            if self.preamble==0 and round_i==0:
                all_solns, stats = self.local_train(round_i, self.clients)
                self.update_local_gradient(self.latest_model, all_solns, self.clients)
                latest_model = self.aggregate(all_solns)
                self.update_global_gradient(latest_model)
                self.update_norms(all_solns, self.latest_model, self.clients)

            if round_i < self.preamble:
                selected_clients = self.select_clients(seed=round_i)
            else:
                selected_clients,_ = self.select_clients_with_prob(solns, selected_clients)

            client_epoch = self.adjust_epoch_gai(selected_clients,round_i,Acc,options)
            selected_clients,client_epoch, repeated_times = self.del_clients(selected_clients,client_epoch,None,round_i)
            selected_clients_index = []
            for i in selected_clients:
                selected_clients_index.append(i.cid)
            solns, stats = self.local_train(round_i, selected_clients,client_epoch)
            """
            update_local_gradient
            """
            self.update_local_gradient(pice,self.latest_model,solns,selected_clients)
            self.metrics.extend_commu_stats(round_i, stats)

            """
            Update latest model
            """
            # Update latest model
            if round_i <600:
                latest_model = self.aggregate(solns, round_i, pice)
            else:
                latest_model = self.aggregate_lambda(solns, stats, Acc, round_i)
                
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

            self.last_time = [0 for _ in range(self.num_clients)]
            
            for c in selected_clients:
                my_dict[c.cid].append(round_i)
                self.repeat_time[c.cid] += 1
                self.last_time[c.cid] += 1

            print(self.repeat_time)
        # Test final model on train cifar-10-batches-py
        self.test_latest_model_on_traindata(self.num_round)
        acc = self.test_latest_model_on_evaldata(self.num_round)
        # Save tracked information
        self.metrics.write()
        return acc

    def update_norms(self,solns,latest_model,select_clients):

        self.norms = [0. for _ in range(len(self.clients))]
        for c, (num_sample, local_solution) in zip(select_clients,solns):
            norm = np.linalg.norm((local_solution-latest_model).cpu())
            self.norms[c.cid] = norm

    def compute_prob(self):
        probs = []
        for c in self.clients:
            probs.append(len(c.train_data))
        return np.array(probs)/sum(probs)

    def find_smallest_indices(self, lst, n):
        indexed_lst = list(enumerate(lst))
        indexed_lst.sort(key=lambda x: x[1])
        smallest_indices = [index for index, value in indexed_lst[:n]]
        return smallest_indices

    def select_clients_with_prob1(self,pre_selected_clients=None, all_solns=None):
        print("FPS sample")
        def dis(x, y):
            dist = -self.get_cos_similar(x,y)
            return dist
        select_clients = []
        init = self.find_smallest_indices(self.repeat_time,1)
        init_point = np.random.randint(0, self.num_clients)
        selected = set()
        for i in init:
            selected.add(i)
        solns = []
        self.d = []
        self.d.append(-1.0)

        max_dis = -999999
        p = 0
        for j in range(self.num_clients):
            if j not in selected:
                d = 0
                for k in selected:
                    d += dis(self.gradients[k], self.gradients[j])
                if d > max_dis:
                    max_dis = d
                    p = j
        selected.add(p)
        self.d.append(max_dis)
        new_gradients = (self.gradients[p] + self.gradients[init[0]])/2


        for i in range(self.clients_per_round - 2 ):
            max_dis = -999999
            p = 0
            for j in range(self.num_clients):
                if j not in selected:
                    d = 0
                    d += dis(new_gradients, self.gradients[j])
                    if d > max_dis:
                        max_dis = d
                        p = j
                        new_gradients = (new_gradients + self.gradients[p]) / 2
            selected.add(p)
            self.d.append(max_dis)

        for p,i in enumerate(selected):
            select_clients.append(self.clients[i])

        select_clients = select_clients[:self.clients_per_round]
        return select_clients,solns



    def select_clients_with_prob(self, selected_clients, pre_selected_clients=None, all_solns=None):
        print("FPS sample")
        ss = []

        for num_sample, id, local_solution in selected_clients:
            ss.append(id)
        sss = random.sample(ss, int(len(selected_clients) / 2))
        while self.are_lists_equal_unordered(sss,ss):
            sss = random.sample(ss, int(len(selected_clients)))   
        def dis(x, y):
            dist = -self.get_cos_similar(x,y)
            return dist
        selected = set()
        selected_old = set()
        for i in ss:
            selected_old.add(i)
        for i in sss:
            selected.add(i)
        solns = []
        self.d = []
        select_clients = []
        self.d.append(-1.0)
        for i in range(self.clients_per_round - int(len(selected_clients) / 2)):
            max_dis = -999999
            p = 0
            for j in range(self.num_clients):
                if j not in selected and j not in selected_old:
                    d = 0
                    for k in selected:
                        d += dis(self.gradients[k], self.gradients[j])
                    if d > max_dis:
                        max_dis = d
                        p = j
            selected.add(p)
            self.d.append(max_dis)

        print("selected_clients!!!",selected)

        for p,i in enumerate(selected):
            select_clients.append(self.clients[i])
        select_clients = select_clients[:self.clients_per_round]
        return select_clients,solns

    def are_lists_equal_unordered(self, list1, list2):
        return Counter(list1) == Counter(list2)
    def aggregate(self, solns, round_i, pice):
        averaged_solution = torch.zeros_like(self.latest_model)
        accum_sample_num = 0
        for num_sample, id, local_solution in solns:
            accum_sample_num += 1
            averaged_solution += 1 * (local_solution - self.latest_model)
        averaged_solution /= accum_sample_num
        if round_i > 100 and round_i % 5 != 0:
            flaction_parament = torch.zeros_like(self.latest_model)
            flaction_parament[pice[self.flag]:pice[self.flag + 1]] = 1
            averaged_solution = averaged_solution * flaction_parament
            self.parament_count += pice[self.flag + 1] - pice[self.flag]
        else:
            self.parament_count += pice[-1]
        print("num_parament:",self.parament_count)
        averaged_solution += self.latest_model
        # averaged_solution += (1-accum_sample_num/self.all_train_data_num) * self.latest_model
        return averaged_solution.detach()
    def aggregate_1(self, solns,selected_clients, stats, **kwargs):
        averaged_solution = torch.zeros_like(self.latest_model)
        sn = 0.
        if self.simple_average:
            for i, (_,id,local_solution) in enumerate(solns):
                averaged_solution += (local_solution-self.latest_model)
            averaged_solution /= len(solns)
        averaged_solution += self.latest_model
        return averaged_solution.detach()
    def parament_select(self, pice,selected_clients_index):
        parament_flag = []

        for i in range(len(pice)-1):
            avg = []
            for j in selected_clients_index:
                avg.append(self.get_cos_similar(self.raw_gradients[j][pice[i]:pice[i+1]],self.raw_global_update[pice[i]:pice[i+1]]))
            parament_flag.append(sum(avg)/len(avg))
        return parament_flag.index(min(parament_flag))
    def aggregate_lambda(self, solns, stats, Acc, round_i):
        def find_loss(id, temp):
            for i in temp:
                if id == i[0]:
                    return i[1]
                else:
                    continue
        select_num = int(self.clients_per_round)
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
        for i in range(int(select_num)):
            set_loss.add(sorted_temp[i][0])
            set_gradient.add(sorted_model_parament[i][0])
        intersection_set = set_loss.intersection(set_gradient)
        select_aggregate = list(intersection_set)
        mmax = 0
        def find_loss(id, temp):
            for i in temp:
                if id == i[0]:
                    return i[1]
                else:
                    continue
        for num_sample, id, local_solution in solns:

            if id in select_aggregate:

                accum_sample_num +=(find_loss(id, temp) + (1- find_loss(id, model_parament)))
                averaged_solution += (find_loss(id, temp) + (1- find_loss(id, model_parament))) * (local_solution - self.latest_model)
        
        averaged_solution /= accum_sample_num
        averaged_solution += self.latest_model
        averaged_solution = averaged_solution
        # averaged_solution += (1-accum_sample_num/self.all_train_data_num) * self.latest_model
        return averaged_solution.detach()
    def aggregate_fps(self, solns, selected_clients,stats, **kwargs):
        aggregate_client = []
        averaged_solution = torch.zeros_like(self.latest_model)
        for i in stats:
            aggregate_client.append(i['id'])
        aggregate_info = []
        sum_cos = 0
        loss_max = stats[0]['loss']
        loss_min = stats[0]['loss']
        for i in stats:
            aggregate_info.append([i['id'], i['loss'],self.get_cos_similar(self.raw_gradients[i['id']],self.raw_global_update)])
            if i['loss'] > loss_max:
                loss_max = i['loss']
            if i['loss'] < loss_min:
                loss_min = i['loss']
        # print(aggregate_info)
        for i, j in enumerate(aggregate_info):
            aggregate_info[i][1] = (loss_max - j[1]) / (loss_max - loss_min)
        for i,j in enumerate(aggregate_info):
            sum_cos += aggregate_info[i][2] + j[1]
        for i, (_,id,local_solution) in enumerate(solns):
            averaged_solution += local_solution*(aggregate_info[i][2] + aggregate_info[i][1]) #*(1.0/d[i])
        averaged_solution = averaged_solution / (sum_cos)
        # print(averaged_solution)
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
