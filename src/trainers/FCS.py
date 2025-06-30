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

criterion = torch.nn.CrossEntropyLoss()


class FPSTrainer(BaseTrainer):
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

        super(FPSTrainer, self).__init__(options, dataset, worker=worker)

        self.last_time = [0 for _ in range(self.num_clients)]

    def train(self,options):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))
        Acc = []
        self.latest_model = self.worker.get_flat_model_params().detach()

        solns = None
        my_dict = {i: [] for i in range(0, len(self.clients) + 1)}
        for round_i in range(self.num_round):

            Acc.append(self.test_latest_model_on_evaldata(round_i))
            print(Acc)
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

            if round_i<self.preamble:
                selected_clients = self.select_clients(seed=round_i)
            else:
                selected_clients,_ = self.select_clients_with_prob(solns)

            # selected_clients, solns = self.select_clients_with_prob()

            # selected_clients, solns = self.select_clients_with_prob(pre_selected_clients,all_solns)

            client_epoch = self.adjust_epoch(selected_clients,round_i)
            selected_clients,client_epoch, repeated_times = self.del_clients(selected_clients,client_epoch,None,round_i)

            solns, stats = self.local_train(round_i, selected_clients)
            """
            update_local_gradient
            """
            pice=[]
            self.update_local_gradient(pice,self.latest_model,solns,selected_clients)
            # self.update_norms(solns, self.latest_model, selected_clients)
            # Track communication cost
            self.metrics.extend_commu_stats(round_i, stats)

            """
            Update latest model
            """
            # Update latest model
            if round_i <600:
                latest_model = self.aggregate_1(solns,selected_clients, stats)
            else:
                # latest_model = self.aggregate_lambda1(solns,selected_clients, stats)
                # latest_model = self.aggregate_1(solns,selected_clients, stats)
                latest_model = self.aggregate_lambda(solns, stats, Acc, round_i)
                

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

            self.last_time = [0 for _ in range(self.num_clients)]
            
            for c in selected_clients:
                my_dict[c.cid].append(round_i)
                self.repeat_time[c.cid] += 1
                self.last_time[c.cid] += 1

        self.test_latest_model_on_traindata(self.num_round)
        acc = self.test_latest_model_on_evaldata(self.num_round)
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



    def select_clients_with_prob(self,pre_selected_clients=None, all_solns=None):
        print("FPS sample")
        def dis(x, y):
            dist = -self.get_cos_similar(x,y)
            return dist
        select_clients = []

        init = self.find_smallest_indices(self.repeat_time,1)
        init_point = np.random.randint(0, self.num_clients)

        selected = set()
        selected.add(init_point)

        ds = []

        # pre_index = [c.cid for c in pre_selected_clients]
        def G(s):
            d = 0
            for k in range(self.num_clients):
                    md = sys.maxsize
                    for i in s:
                        md = min(md,np.linalg.norm(self.gradients[k]-self.gradients[i]))
                    d += md
            return d

        solns = []
        self.d = []
        self.d.append(-1.0)
        for i in range(self.clients_per_round - 1 ):
            max_dis = -999999
            p = 0
            for j in range(self.num_clients):
                if j not in selected:
                    d = 0
                    for k in selected:
                        d += dis(self.gradients[k], self.gradients[j])
                    # d = d*(1+np.log(self.last_time[j]))

                    if d > max_dis:
                        max_dis = d
                        p = j
            selected.add(p)
            self.d.append(max_dis)

        print(selected)

        for p,i in enumerate(selected):
            select_clients.append(self.clients[i])

        select_clients = select_clients[:self.clients_per_round]
        return select_clients,solns


    def aggregate_1(self, solns,selected_clients, stats, **kwargs):
        averaged_solution = torch.zeros_like(self.latest_model)
        if self.simple_average:
            for i, (_,id,local_solution) in enumerate(solns):

                averaged_solution += (local_solution-self.latest_model)#*(1.0/d[i])
            averaged_solution /= len(solns)
        averaged_solution += self.latest_model
        return averaged_solution.detach()
    
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
        # print(sorted_model_parament)
        # print(sorted_temp)
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
        return averaged_solution.detach()
    def alpha(self, accuracy_prev):
        accuracy_min = 0.05
        accuracy_max = 0.93

        α_min = 0.07
        α_max = 0.66


        k = 9.0

        normalized_accuracy = (accuracy_prev - accuracy_min) / (accuracy_max - accuracy_min)


        α_exp = math.exp(k * normalized_accuracy)

        α_min_exp = math.exp(k * 0) 
        α_max_exp = math.exp(k * 1) 


        α = α_min + (α_max - α_min) * (α_exp - α_min_exp) / (α_max_exp - α_min_exp)
        return α
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

        for i, j in enumerate(aggregate_info):
            aggregate_info[i][1] = (loss_max - j[1]) / (loss_max - loss_min)
        for i,j in enumerate(aggregate_info):
            sum_cos += aggregate_info[i][2] + j[1]
        for i, (_,id,local_solution) in enumerate(solns):
            averaged_solution += local_solution*(aggregate_info[i][2] + aggregate_info[i][1]) #*(1.0/d[i])
        averaged_solution = averaged_solution / (sum_cos)
        return averaged_solution.detach()
        
        









        sorted_data = sorted(stats, key=lambda x: x['loss'])
        aggregate_client = []
        num_select_client = len(selected_clients)
        sorted_data = sorted_data[:int(num_select_client * 0.8)]
        for i in sorted_data:
            aggregate_client.append(i['id'])
        index_slons = []
        for i, j in enumerate(stats):
            if j['id'] in aggregate_client:
                index_slons.append(i)
        new_slons = []
        for i in range(num_select_client):
            if i in index_slons:
                new_slons.append(solns[i])
        new_dict = {}
        s = []
        for i in index_slons:
            new_dict['id'] = stats[i]['id']
            new_dict['slons'] = solns[i][1]
            new_dict['cos'] = self.get_cos_similar(self.gradients[stats[i]['id']],self.global_update)
            s.append(new_dict)
            new_dict = {}
        for i, (_,local_solution) in enumerate(new_slons):
            averaged_solution += (local_solution-self.latest_model)#*(1.0/d[i])
        averaged_solution /= len(solns)
        averaged_solution += self.latest_model
        return averaged_solution.detach()

        # if self.simple_average:
        #     # print(solns)
        #     # ag = np.average([t[1].numpy() for t in solns],axis=0)
        #     for i, (_,local_solution) in enumerate(solns):
        #         # w = 1.0/self.get_cos_similar(local_solution,ag)

        #         averaged_solution += (local_solution-self.latest_model)#*(1.0/d[i])
        #     averaged_solution /= len(solns)
        # averaged_solution += self.latest_model
        # # averaged_solution = from_numpy(averaged_solution, self.gpu)

