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
class FedAvg6Trainer(BaseTrainer):
    """
    uniform Scheme
    """
    def __init__(self, options, dataset):
        self.model = choose_model(options)
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(f"{name}: {param.numel()} parameters")
        # exit()
        self.move_model_to_gpu(self.model, options)
        if options['optim'] == 'GD':
            print("GGGGGGGGGGGGGD")
            self.optimizer = GD(self.model.parameters(), lr=options['lr'], weight_decay=options['wd'])
        elif options['optim'] == 'Adam':
            print("Adam!!!!!!!!!!!!")
            self.optimizer = optim.Adam(self.model.parameters(), lr=options['lr'], weight_decay=5e-4)
        elif options['optim'] == 'SGD':
            print("SGD!!!!!!!!!!!!!!")
            self.optimizer = optim.SGD(self.model.parameters(), lr=options['lr'], momentum=0.9, weight_decay=5e-4)
        elif options['optim'] == 'AdamW':
            print("AdamWWWW!!!!!!!!!!!!")
            optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-2)       
        #self.optimizer = optim.Adam(model.parameters(), lr=options['lr'])

        self.num_epoch = options['num_epoch']
        worker = LrdWorker(self.model, self.optimizer, options)
        super(FedAvg6Trainer, self).__init__(options, dataset, worker=worker)

    def train(self,options):

        print('>>> Select {} clients per round \n'.format(self.clients_per_round))
        Acc = []
        self.flag = []

        # Fetch latest flat model parameter
        self.latest_model = self.worker.get_flat_model_params().detach()
        max_acc = 0
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
        for round_i in range(self.num_round):
            flag_list.append(self.flag)
            print(flag_list)
            # Test latest model on train cifar-10-batches-py
            # self.test_latest_model_on_traindata(round_i)
            Acc.append(self.test_latest_model_on_evaldata(round_i))
            print(Acc[round_i])
            print(Acc)
            if Acc[round_i] > max_acc:
                max_acc = Acc[round_i]
                best_model = self.latest_model
            print(max_acc)
            print(best_model)
            selected_clients = self.select_clients(seed=round_i)
            selected_clients_index = []
            for i in selected_clients:
                selected_clients_index.append(i.cid)

            solns, stats = self.local_train(round_i, selected_clients)

            self.update_local_gradient(pice,self.latest_model,solns,selected_clients)
            # Track communication cost
            self.metrics.extend_commu_stats(round_i, stats)
            if round_i <= 500 :
                latest_model = self.aggregate(solns, round_i, pice)
            else:
                #latest_model = self.aggregate(solns)
                latest_model = self.aggregate_lambda(solns,stats,Acc,round_i,self.clients_per_round)
            if options['model'] == 'cnn':    
                if round_i == 99:
                    print("last parament count:",self.parament_count)
                    print("$$$$$$$$$$$$:::",self.parament_count*4*options['clients_per_round']/50/1048576,"MB")
            else:
                if round_i == 199:
                    print("last parament count:",self.parament_count)
                    print("$$$$$$$$$$$$:::",self.parament_count*4*options['clients_per_round']/90/1048576,"MB")
            self.update_global_gradient(latest_model)
            if round_i >= 100 and round_i % 5 == 0:
                self.flag = self.parament_select(pice,selected_clients_index)
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
        # Test final model on train cifar-10-batches-py
        self.test_latest_model_on_traindata(self.num_round)
        self.test_latest_model_on_evaldata(self.num_round)

        print(self.repeat_time)
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
        def two_smallest_indices(lst,k):
            return [index for index, _ in sorted(enumerate(lst), key=lambda x: x[1])[:k]]
        return two_smallest_indices(parament_flag,3)
        # return parament_flag.index(min(parament_flag))
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
            for i in self.flag:
                flaction_parament[pice[i]:pice[i + 1]] = 1
                self.parament_count += pice[i + 1] - pice[i]
            averaged_solution = averaged_solution * flaction_parament
            
        else:
            pass
        print("num_parament:",self.parament_count)
        averaged_solution += self.latest_model
        # averaged_solution += (1-accum_sample_num/self.all_train_data_num) * self.latest_model
        return averaged_solution.detach()
    def aggregate_lambda(self, solns, stats, Acc, round_i,clients_per_round):
        select_num = clients_per_round
        averaged_solution = torch.zeros_like(self.latest_model)
        accum_sample_num = 0
        temp = []
        for i in stats:
            temp.append([i['id'], i['loss']])
        sorted_temp = sorted(temp, key = lambda x: x[1], reverse = True)
        model_parament = []
        gradient_l2 = []

        for i in temp:
            gradient_l2.append([i[0], np.linalg.norm(self.gradients[i[0]])])
            model_parament.append([i[0], self.get_cos_similar(self.gradients[i[0]], self.global_update)])
        sorted_model_parament = sorted(model_parament, key = lambda x: x[1], reverse = True)
        # print(sorted_model_parament)
        # print(sorted_temp)
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
            # if (id in select_aggregate) and (id in select_aggregate_origin):
            #     accum_sample_num += num_sample*1.5
            #     averaged_solution += num_sample* 1.5 * local_solution
            if id in select_aggregate:
                # accum_sample_num += find_loss(id, temp) + (1 - find_loss(id, model_parament))
                # averaged_solution += (find_loss(id, temp) + (1 - find_loss(id, model_parament))) * (local_solution - self.latest_model)
                # accum_sample_num +=(find_loss(id, temp)*3 + (1- find_loss(id, model_parament)))
                # averaged_solution += (find_loss(id, temp)*3 + (1- find_loss(id, model_parament))) * (local_solution - self.latest_model)
                # accum_sample_num += 1
                # averaged_solution += 1 * (local_solution - self.latest_model)
                accum_sample_num +=(2*find_loss(id, temp) + (1- find_loss(id, model_parament)))
                averaged_solution += (2*find_loss(id, temp) + (1- find_loss(id, model_parament))) * (local_solution - self.latest_model)

        # accum_sample_num += mmax
        # averaged_solution += mmax * self.latest_model
        averaged_solution /= accum_sample_num
        averaged_solution += self.latest_model
        α = self.alpha(Acc[round_i])
        α = 0
        print(α, "1111111111111")
        averaged_solution = (1 - α) * averaged_solution + α * self.latest_model
        # averaged_solution += (1-accum_sample_num/self.all_train_data_num) * self.latest_model
        return averaged_solution.detach()
    def aggregate_lambda1(self, solns, stats, Acc, round_i):
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
        # print(sorted_model_parament)
        # print(sorted_temp)
        set_loss = set()
        set_gradient = set()
        for i in range(select_num):
            set_loss.add(sorted_temp[i][0])
            set_gradient.add(sorted_model_parament[i][0])
        intersection_set = set_loss.intersection(set_gradient)
        select_aggregate = list(intersection_set)
        print(len(select_aggregate),"22222222222")
        select_aggregate_origin = copy.deepcopy(select_aggregate)
        if len(select_aggregate) < select_num:
            for i in sorted_temp:
                if i[0] not in select_aggregate:
                    select_aggregate.append(i[0])
                if len(select_aggregate) == select_num:
                    break
        mmax = 0
        for num_sample, id, local_solution in solns:
            # if (id in select_aggregate) and (id in select_aggregate_origin):
            #     accum_sample_num += num_sample*1.5
            #     averaged_solution += num_sample* 1.5 * local_solution
            if id in select_aggregate:
                accum_sample_num += 1
                averaged_solution += 1 * (local_solution - self.latest_model) 
        # accum_sample_num += mmax
        # averaged_solution += mmax * self.latest_model
        averaged_solution /= accum_sample_num
        averaged_solution += self.latest_model
        α = self.alpha(Acc[round_i])
        α = 0
        print(α, "1111111111111")
        averaged_solution = (1 - α) * averaged_solution + α * self.latest_model
        # averaged_solution += (1-accum_sample_num/self.all_train_data_num) * self.latest_model
        return averaged_solution.detach()
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

