import math
import random

import numpy as np
import torch
import time
from src.models.client import Client
from src.utils.worker_utils import Metrics
from src.models.worker import Worker
from sklearn.random_projection import SparseRandomProjection
from queue import Queue
import heapq
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from random import sample
import pandas as pd
from numpy.random import uniform
from collections import defaultdict
from src.utils.worker_utils import read_data



class BaseTrainer(object):
    def __init__(self, options, dataset, model=None, optimizer=None, name='', worker=None):
        if model is not None and optimizer is not None:
            self.worker = Worker(model, optimizer, options)
        elif worker is not None:
            self.worker = worker
        else:
            raise ValueError("Unable to establish a worker! Check your input parameter!")
        print('>>> Activate a worker for training')
        self.data_name = options['dataset']
        self.parament_count = 0
        self.u_count = []
        self.gpu = options['gpu']
        self.batch_size = options['batch_size']
        self.all_train_data_num = 0
        self.clients = self.setup_clients(dataset)
        assert len(self.clients) > 0
        print('>>> Initialize {} clients in total'.format(len(self.clients)))

        self.num_round = options['num_round']
        self.clients_per_round = options['clients_per_round']
        self.num_clients = len(self.clients)
        self.clients_per_round = min(self.clients_per_round, self.num_clients)
        self.eval_every = options['eval_every']
        self.simple_average = not options['noaverage']
        print('>>> Weigh updates by {}'.format(
            'simple average' if self.simple_average else 'sample numbers'))

        # Initialize system metrics
        self.name = '_'.join([name, f'wn{self.clients_per_round}', f'tn{len(self.clients)}'])
        self.metrics = Metrics(self.clients, options, self.name)
        self.print_result = not options['noprint']
        self.latest_model = self.worker.get_flat_model_params()
        self.max_test_acc = [0., 0]

        self.num_epoch = options['num_epoch']
        self.raw_num_epoch = options['num_epoch']
        self.num_epochs = []

        self.rg = options['real_gradient']
        self.ce = options['decrease_E']
        self.de = options['dynamic_E']

        self.ie = False

        self.compress_dim = options['compress_dim']
        self.momentum = options['momentum']

        self.min_similarity = options['min_similarity']

        self.preamble = options['preamble'] #10 # if self.gpu else 5
        self.repeat_time = [1 for _ in range(self.num_clients)]

        self.raw_gradients = np.array([0. * (np.array(torch.rand_like(self.latest_model).cpu())) for _ in
                          range(len(self.clients))])

        self.gradients = np.array([np.zeros(self.compress_dim,float) for _ in
                          range(len(self.clients))])

        self.g_mask = 0. * (np.array(torch.rand_like(self.latest_model).cpu()))

        self.raw_global_update = np.array([0. for _ in range(len(self.raw_gradients[0]))])
        self.global_update = np.array([0. for _ in range(self.compress_dim)])
        self.norms = [0. for i in range(len(self.clients))]
        self.global_gradients = []

        self.projection = options['projection']
        self.transformer = SparseRandomProjection()

        self.vars = []

        self.dc = options['del_client']
        self.dc_times = 0

        self.um = False

        self.random_gradient = options['random_gradient']

        self.sim_var = []
        self.accs = []
        self.tt = 0

        self.scales = []

        self.lables = []

    @staticmethod
    def move_model_to_gpu(model, options):
        if 'gpu' in options and (options['gpu'] is True):
            device = 0 if 'device' not in options else options['device']
            torch.cuda.set_device(device)
            torch.backends.cudnn.enabled = True
            model.cuda()
            print('>>> Use gpu on device {}'.format(device))
        else:
            print('>>> Don not use gpu')

    def setup_clients(self, dataset):
        """Instantiates clients based on given train and test cifar-10-batches-py directories

        Returns:
            all_clients: List of clients
        """
        
        users, groups, train_data, test_data = dataset

        if len(groups) == 0:
            groups = [None for _ in users]

        all_clients = []
        for user, group in zip(users, groups):
            if isinstance(user, str) and len(user) >= 5:
                user_id = int(user[-5:])
            else:
                user_id = int(user)
            self.all_train_data_num += len(train_data[user])
            c = Client(user_id, group, train_data[user], test_data[user], self.batch_size, self.worker)
            all_clients.append(c)
            # if len(all_clients)>=10:
            #     break
        return all_clients

    def train(self):
        """The whole training procedure

        No returns. All results all be saved.
        """
        raise NotImplementedError

    def select_clients(self, seed=1):
        """Selects num_clients clients weighted by number of samples from possible_clients

        Args:
            1. seed: random seed
            2. num_clients: number of clients to select; default 20
                note that within function, num_clients is set to min(num_clients, len(possible_clients))

        Return:
            list of selected clients objects
        """
        print("fedavg5/fedavg9  uniform sample")
        num_clients = min(self.clients_per_round, len(self.clients))
        np.random.seed(seed)
        return np.random.choice(self.clients, num_clients, replace=False).tolist()

    def local_train(self, round_i, selected_clients, client_epochs=None,**kwargs):
        """Training procedure for selected local clients

        Args:
            round_i: i-th round training
            selected_clients: list of selected clients

        Returns:
            solns: local solutions, list of the tuple (num_sample, local_solution)
            stats: Dict of some statistics
        """
        solns = []  # Buffer for receiving client solutions
        stats = []  # Buffer for receiving client communication costs
        for i, c in enumerate(selected_clients, start=1):
            # Communicate the latest model
            c.set_flat_model_params(self.latest_model)
            # Solve minimization locally
            soln, stat = c.local_train( num_epoch=client_epochs[i-1] if client_epochs is not None else None, latest_model =self.latest_model)
            if self.print_result:
                print("Round: {:>2d} | CID: {: >3d} ({:>2d}/{:>2d})| "
                      "Param: norm {:>.4f} ({:>.4f}->{:>.4f})| "
                      "Loss {:>.4f} | Acc {:>5.2f}% | Time: {:>.2f}s".format(
                       round_i, c.cid, i, self.clients_per_round,
                       stat['norm'], stat['min'], stat['max'],
                       stat['loss'], stat['acc'], stat['time']))

            # Add solutions and stats
            solns.append(soln)
            stats.append(stat)

        return solns, stats

    def aggregate(self, solns, round_i, pice):
        k = 100
        averaged_solution = torch.zeros_like(self.latest_model)
        accum_sample_num = 0
        for num_sample, id, local_solution in solns:
            accum_sample_num += 1
            averaged_solution += 1 * (local_solution - self.latest_model)
        averaged_solution /= accum_sample_num
        if round_i > k and round_i % 5 != 0:
            flaction_parament = torch.zeros_like(self.latest_model)
            flaction_parament[pice[self.flag]:pice[self.flag + 1]] = 1
            averaged_solution = averaged_solution * flaction_parament
            self.parament_count += pice[self.flag + 1] - pice[self.flag]
        else:
            pass
        print("num_parament:",self.parament_count)
        averaged_solution += self.latest_model
        # averaged_solution += (1-accum_sample_num/self.all_train_data_num) * self.latest_model
        return averaged_solution.detach()

    def test_latest_model_on_traindata(self, round_i):
        # Collect stats from total train cifar-10-batches-py
        begin_time = time.time()
        stats_from_train_data = self.local_test(use_eval_data=False)

        # Record the global gradient
        model_len = len(self.latest_model)
        global_grads = np.zeros(model_len)
        num_samples = []
        local_grads = []

        for c in self.clients:
            (num, client_grad), stat = c.solve_grad()
            local_grads.append(client_grad)
            num_samples.append(num)
            global_grads += client_grad * num
        global_grads /= np.sum(np.asarray(num_samples))
        stats_from_train_data['gradnorm'] = np.linalg.norm(global_grads)

        # Measure the gradient difference
        difference = 0.
        for idx in range(len(self.clients)):
            difference += np.sum(np.square(global_grads - local_grads[idx]))
        difference /= len(self.clients)
        stats_from_train_data['graddiff'] = difference
        end_time = time.time()

        self.metrics.update_train_stats(round_i, stats_from_train_data)
        if self.print_result:
            print('\n>>> Round: {: >4d} / Acc: {:.3%} / Loss: {:.4f} /'
                  ' Grad Norm: {:.4f} / Grad Diff: {:.4f} / Time: {:.2f}s'.format(
                   round_i, stats_from_train_data['acc'], stats_from_train_data['loss'],
                   stats_from_train_data['gradnorm'], difference, end_time-begin_time))
            print('=' * 102 + "\n")
        return global_grads

    def test_latest_model_on_evaldata(self, round_i):
        begin_time = time.time()
        stats_from_eval_data = self.local_test(use_eval_data=True)
        end_time = time.time()

        if self.print_result and round_i % self.eval_every == 0:
            print('= Test = round: {} / acc: {:.3%} / '
                  'loss: {:.4f} / Time: {:.2f}s'.format(
                   round_i, stats_from_eval_data['acc'],
                   stats_from_eval_data['loss'], end_time-begin_time))
            print('=' * 102 + "\n")
        if self.max_test_acc[0] < stats_from_eval_data['acc']:
            self.max_test_acc = [stats_from_eval_data['acc'], round_i]
        print("best acc {} in round {}".format(self.max_test_acc[0],self.max_test_acc[1]))
        self.metrics.update_eval_stats(round_i, stats_from_eval_data)
        self.accs.append(stats_from_eval_data['acc'])
        return stats_from_eval_data['acc']

    def local_test(self, use_eval_data=True):
        assert self.latest_model is not None
        self.worker.set_flat_model_params(self.latest_model)

        num_samples = []
        tot_corrects = []
        losses = []
        for c in self.clients:
            tot_correct, num_sample, loss = c.local_test(use_eval_data=use_eval_data)

            tot_corrects.append(tot_correct)
            num_samples.append(num_sample)
            losses.append(loss)
        print(tot_corrects)
        ids = [c.cid for c in self.clients]
        groups = [c.group for c in self.clients]

        stats = {'acc': sum(tot_corrects) / sum(num_samples),
                 'loss': sum(losses) / sum(num_samples),
                 'num_samples': num_samples, 'ids': ids, 'groups': groups}

        return stats

    def normalization(self,data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

    def standardization(self,data):
        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)
        return (data - mu) / sigma

    def update_local_gradient(self,pice,last_model,solns,selected_clients, clients_epoch=None, round_i=None):
        gradients = []
        for i,(c, (num_sample,id, local_solution)) in enumerate(zip(selected_clients,solns)):
            gradient = np.array((local_solution-last_model).cpu())
            gradients.append(gradient)
            if self.momentum:
                self.raw_gradients[c.cid] = 0.1* gradient+0.9*self.raw_gradients[c.cid]
            else:
                self.raw_gradients[c.cid] = gradient    
        self.gradients = self.raw_gradients[:, -self.compress_dim:]

        gradients = np.array(gradients)
        var = np.var(gradients, axis=0)
        if self.um and round_i<=int(self.preamble/4):
            for i in range(len(var)):
                self.g_mask[i] += var[i]
        self.vars.append(np.sum(var))
        print("avg model vars ", np.average(self.vars))

    def update_global_gradient(self,latest_model, round_i=None):
        if self.momentum:
            self.raw_global_update = 0.1 * (latest_model - self.latest_model).cpu() + 0.9 * self.raw_global_update
        else:
            self.raw_global_update = (latest_model - self.latest_model).cpu()
            self.global_gradients.append(self.raw_global_update.numpy())

        if self.projection:
            self.raw_global_update = np.expand_dims(self.raw_global_update,0)
            # print(self.raw_gradients.shape, self.raw_global_update.shape)
            t = np.concatenate((self.raw_gradients,self.raw_global_update),axis=0)
            # print(t.shape)
            t_new = self.transformer.fit_transform(t)
            self.gradients = t_new[:100]
            self.global_update = t_new[-1]
        elif self.um:
            if round_i<int(self.preamble/4):
                return
            elif round_i==int(self.preamble/4):
                t = heapq.nlargest(100,self.g_mask)[-1]
                for i,v in enumerate(self.g_mask):
                    if v<t:
                        self.g_mask[i] = 0
                    else:
                        self.g_mask[i] = 1

            t = np.zeros((100,100))
            j = 0
            for i,v in enumerate(self.g_mask):
                if v==1:
                    t[:,j] = self.raw_gradients[:,i]
                    j += 1
            self.gradients = self.gradients - self.global_update + t
            self.global_update = np.average(self.gradients, axis=0)
        else:
            self.gradients = self.raw_gradients[:, -self.compress_dim:]
            self.global_update = np.array(self.raw_global_update)[-self.compress_dim:]


    def get_cos_similar(self, v1, v2):
        num = float(np.dot(v1, v2))
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        return 0.5 + 0.5 * (num / denom) if denom != 0 else 0

    def allocation_amount(self,num_people, amount):
        a = [np.random.uniform(0, amount) for i in range(num_people-1)]
        a.append(0)
        a.append(amount)
        a.sort()
        b = [max(a[i + 1] - a[i],0.1) for i in range(num_people)]  
        b = np.array(b)
        return b

    def hopkins_statistic(self,x):
        d = x.shape[1]
        n = len(x)
        m = int(0.5 * n)
        nbrs = NearestNeighbors(n_neighbors=1).fit(x.values)
        rand_x = sample(range(0, n), m)
        ujd = []
        wjd = []

        for j in range(0, m):
            u_dist, _ = nbrs.kneighbors(uniform(np.min(x, axis=0), np.max(x, axis=0), d).reshape(1, -1), 2)
            ujd.append(u_dist[0][1])
            w_dist, _ = nbrs.kneighbors(x.iloc[rand_x[j]].values.reshape(1, -1), 2)
            wjd.append(w_dist[0][1])
        h = sum(ujd) / (sum(ujd) + sum(wjd))
        return h

    def hopkins_statistic_2(self, index):
        out = []

        for i in range(len(self.clients)):
            if i not in index:
                out.append(i)

        idis = []
        odis = []
        for i in index[:5]:
            mini = 0.
            for j in index:
                if i == j:
                    continue
                t = self.get_cos_similar(self.gradients[i], self.gradients[j])
                if t != 0:
                    mini = max(mini, t)
            idis.append(mini)
            mino = 0.
            for j in out:
                t = self.get_cos_similar(self.gradients[i], self.gradients[j])
                if t != 0:
                    mino = max(mino, t)
            odis.append(mino)
        print(odis, idis)
        h = sum(odis) / (sum(odis) + sum(idis))

        return h

    def un_degree(self, index):
        # gradients = np.load("gradients.npy")
        t = []
        n = len(index)
        for i, a in enumerate(index):
            for j, b in enumerate(index[i + 1:]):
                t.append(self.get_cos_similar(self.gradients[a], self.gradients[b]))
        return np.average(t)


    def adjust_epoch(self,selected_clients, round_i):
        # change total epoch number
        if self.ce:
            means = np.average(self.vars)
            pre_gradients = []
            for c in selected_clients:
                pre_gradients.append(self.gradients[c.cid])

            s = 0.
            for g in pre_gradients:
                s += self.get_cos_similar(g,self.global_update)
            s = s/len(pre_gradients)
            print("predict next round cos distance is {}, average cos distance is {}".format(s,np.average(self.vars)))

            num_epoch = self.raw_num_epoch- (s - 0.5)

            self.vars.append(s)
            self.vars = self.vars[-20:]
            if round_i > self.preamble:
                #raw_num_epoch = num_epoch
                self.num_epoch = min(max(self.raw_num_epoch/2.,num_epoch), self.raw_num_epoch*1.2)
                self.worker.num_epoch = min(max(self.raw_num_epoch/2.,num_epoch), self.raw_num_epoch*1.2)
                print("total epoch is changed to ",self.worker.num_epoch)
        self.num_epochs.append(self.worker.num_epoch)
        print("now avg epoch is", np.average(self.num_epochs))
        """
        adjust client epoch allocate
        if epoch_i<threshold it may be delete
        """
        def vector_norm(vector):
            return torch.norm(vector, p=2)
        if (self.de or self.dc) and round_i>=self.preamble:
            print("change client epochs")
            index = [c.cid for c in selected_clients]
            distance = []
            zero_client = []
            l2 = []
            def normalize_to_range(arr, new_min, new_max):
                arr_min = np.min(arr)
                arr_max = np.max(arr)
                normalized_arr = new_min + ((arr - arr_min) * (new_max - new_min) / (arr_max - arr_min))
                return normalized_arr
            for i in index:
                l2.append(np.linalg.norm(self.gradients[i]))
            ss = normalize_to_range(l2,0.9,1.3)
            for k,i in enumerate(index):
                local_solution = self.gradients[i]

                if self.random_gradient:
                    distance.append(
                        max(self.min_similarity, random.random()))
                else:
                    if (local_solution==np.zeros(len(local_solution),float)).all():
                        print(i,"gradient is zero")
                        zero_client.append(i)
                        distance.append(self.min_similarity)
                    else:
                        distance.append(max(self.min_similarity*ss[k],
                                            ss[k] * (1-self.get_cos_similar(local_solution, np.array(self.global_update)))))


            client_epoch = []

            def normalize_exponential(x):
                a = 1
                b = 6 - 1
                return a + b * (math.exp(x) - 1) / (math.e - 1)
            def normalize_linear(x):
                a = 1
                b = 5
                return a + b * x
            for i in range(len(distance)):
                client_epoch.append(((distance[i]) / (sum(distance))) * (1.3*self.num_epoch * (len(selected_clients))))


            if not self.de:
                for i,e in enumerate(client_epoch):
                    if e!=0:
                        client_epoch[i] = self.num_epoch
            # if self.dc:
            #     for i, d in enumerate(dis):
            #         if d == self.min_similarity and n > self.clients_per_round * 0.8:
            #             dis[i] = 0
            #             n -= 1

            print(distance, client_epoch)
            # client_epoch = self.allocation_amount(len(index),self.num_epoch * self.clients_per_round)
            # print(client_epoch)

            return client_epoch
        return None



    def adjust_epoch_old(self,selected_clients, round_i,Acc):
        # change total epoch number
        if self.ce:
            means = np.average(self.vars)
            pre_gradients = []
            for c in selected_clients:
                pre_gradients.append(self.gradients[c.cid])

            s = 0.
            for g in pre_gradients:
                s += self.get_cos_similar(g,self.global_update)
            s = s/len(pre_gradients)
            print("predict next round cos distance is {}, average cos distance is {}".format(s,np.average(self.vars)))

            num_epoch = self.raw_num_epoch- (s - 0.5)
            self.vars.append(s)
            self.vars = self.vars[-20:]
            if round_i > self.preamble:
                self.num_epoch = min(max(self.raw_num_epoch/2.,num_epoch), self.raw_num_epoch*1.2)
                self.worker.num_epoch = min(max(self.raw_num_epoch/2.,num_epoch), self.raw_num_epoch*1.2)
                print("total epoch is changed to ",self.worker.num_epoch)
        self.num_epochs.append(self.worker.num_epoch)
        print("now avg epoch is", np.average(self.num_epochs))

        """
        adjust client epoch allocate
        if epoch_i<threshold it may be delete
        """
        if (self.de or self.dc) and round_i>=self.preamble:
            print("change client epochs")
            # print(self.global_update)
            index = [c.cid for c in selected_clients]
            distance = []

            for i in index:
                local_solution = self.gradients[i]

                if self.random_gradient:
                    distance.append(
                        max(self.min_similarity, random.random()))
                else:
                    if (local_solution==np.zeros(len(local_solution),float)).all():
                        print(i,"gradient is zero")
                        x = 0
                    else:
                        x = self.get_cos_similar(local_solution, self.global_update)
                    distance.append(math.log(1+x))
            for i,t in enumerate(distance):
                if t==0:
                    distance[i]=max(distance)

            client_epoch = []

            """
            adjust
            """
            for i in range(len(distance)):
                x = (distance[i] / sum(distance)) * (self.num_epoch * len(selected_clients))
                client_epoch.append(x)

            if not self.de:
                for i,e in enumerate(client_epoch):
                    if e!=0:
                        client_epoch[i] = self.num_epoch


            return client_epoch
        return None

    def adjust_epoch_gai(self,selected_clients, round_i,Acc,options):
        l2 = []
        index = [c.cid for c in selected_clients]
        def normalize_to_range(arr, new_min, new_max):
                arr_min = np.min(arr)
                arr_max = np.max(arr)
                normalized_arr = new_min + ((arr - arr_min) * (new_max - new_min) / (arr_max - arr_min))
                return normalized_arr

        for i in index:
                l2.append(np.linalg.norm(self.gradients[i]))
        ss = normalize_to_range(l2,0.9,1.1)
        client_epoch =[]
        def acc_standard(acc,round):
            s = sum(acc[round-3:round])
            return s / 3
        if options['model'] == 'cnn':
            if round_i<20:
                u = 2
                print("pre_epoch~~~~~")
                client_epoch=[2]*len(ss)
            else:
                print(acc_standard(Acc,round_i))

                if acc_standard(Acc,round_i) > 0.83:
                    u = 1
                elif acc_standard(Acc,round_i) > 0.50:
                    u = 1.5
                else:
                    u = 2
                for i in range(len(ss)):
                    client_epoch.append((ss[i] / sum(ss)) * (u*len(ss)))
        elif options['model'] == 'cifar' or options['model'] == 'medcnn':
            if round_i<50:
                u = 2
                client_epoch=[2]*len(ss)
            else:
                print(acc_standard(Acc,round_i))

                if acc_standard(Acc,round_i) > 0.55:
                    u = 1
                elif acc_standard(Acc,round_i) > 0.50:
                    u = 1
                elif acc_standard(Acc,round_i) > 0.45:
                    u = 1.3
                else:
                    u = 2
                for i in range(len(ss)):
                    client_epoch.append((ss[i] / sum(ss)) * (u*len(ss)))
                #client_epoch.append(ss[i] * u)
        print(client_epoch)
        self.u_count.append(u)
        return client_epoch
        
    def del_clients(self,selected_clients, client_epoch,repeated_times,round_i ):
        if round_i<self.preamble or not self.dc:
            return selected_clients, client_epoch, repeated_times
        t1 = []
        t2 = []
        t3 = []
        index = [c.cid for c in selected_clients]
        for i,c in enumerate(index):
            # local_solution = self.gradients[c]
            if client_epoch[i]>0.:
                t1.append(selected_clients[i])
                if type(client_epoch) is list:
                    t2.append(client_epoch[i])
                if type(repeated_times) is list:
                    t3.append(repeated_times[i])
            else:
                self.dc_times += 1
        print("delete client {} times".format(self.dc_times))
        return t1,t2,t3