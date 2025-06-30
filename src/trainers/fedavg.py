from src.trainers.base import BaseTrainer
from src.models.model import choose_model
from src.models.worker import LrdWorker
from src.optimizers.gd import GD
import numpy as np
import torch
import copy
import random
import math
import torch.optim as optim
criterion = torch.nn.CrossEntropyLoss()

# uniform sample
class FedAvg5Trainer(BaseTrainer):
    """
    uniform Scheme
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
        super(FedAvg5Trainer, self).__init__(options, dataset, worker=worker)

    def train(self,options):

        print('>>> Select {} clients per round \n'.format(self.clients_per_round))
        Acc = []
        self.flag = -1
        # Fetch latest flat model parameter
        self.latest_model = self.worker.get_flat_model_params().detach()
        max_acc = 0
        index = []
        pice = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # print(f"{name}: {param.numel()} parameters")
                index.append(param.numel())
        for i in range(int(len(index) / 2)):
            pice.append(index[i*2] + index[i*2+1])
        for i in range(1,len(pice)):
            pice[i] = pice[i-1]+pice[i]

        for round_i in range(self.num_round):
            Acc.append(self.test_latest_model_on_evaldata(round_i))
            # Choose K clients prop to cifar-10-batches-py size
            selected_clients = self.select_clients(seed=round_i)
            solns, stats = self.local_train(round_i, selected_clients)
            self.update_local_gradient(pice,self.latest_model,solns,selected_clients)
            # Track communication cost
            self.metrics.extend_commu_stats(round_i, stats)
            latest_model = self.aggregate(solns, round_i, pice)
            self.update_global_gradient(latest_model)
            temp = self.latest_model
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
        self.test_latest_model_on_traindata(self.num_round)
        self.test_latest_model_on_evaldata(self.num_round)

        self.metrics.write()

    def aggregate(self, solns, round_i, pice):
        averaged_solution = torch.zeros_like(self.latest_model)
        accum_sample_num = 0

        for num_sample, id, local_solution in solns:
            accum_sample_num += 1
            averaged_solution += 1 * (local_solution - self.latest_model)
        averaged_solution /= accum_sample_num
        averaged_solution += self.latest_model
        return averaged_solution.detach()
    
