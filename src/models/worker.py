from src.utils.flops_counter import get_model_complexity_info
from src.utils.torch_utils import get_flat_grad, get_state_dict, get_flat_params_from, set_flat_params_to
import torch.nn as nn
import torch
from torch.autograd import Function
import torch.utils.data as data
import torch.nn.functional as F
import copy
criterion = nn.CrossEntropyLoss()
mseloss = nn.MSELoss()


class Worker(object):
    """
    Base worker for all algorithm. Only need to rewrite `self.local_train` method.

    All solution, parameter or grad are Tensor type.
    """
    def __init__(self, model, optimizer, options):
        # Basic parameters
        self.model = model
        # for name, param in model.named_parameters():
        #     print(f"Layer: {name}, Parameter shape: {param.shape}")
        #     print(f"Parameter values: {param}")
        #     print("-" * 50)
        # exit()
        self.optimizer = optimizer
        self.num_epoch = options['num_epoch']
        self.gpu = options['gpu'] if 'gpu' in options else False

        # Setup local model and evaluate its statics
        self.flops, self.params_num, self.model_bytes = \
            get_model_complexity_info(self.model, options['input_shape'], gpu=options['gpu'])

        self.num_round = options['num_round']
        # self.ie = options['incremental_E']
        # if self.ie:
        #     self.num_epoch_seq=[]
        #     for i in range(1,int(self.num_round/2)+1):
        #         self.num_epoch_seq.append(min(i,self.num_epoch*10))
        #         self.num_epoch_seq.append(min(i,self.num_epoch*10))
        #     print(self.num_epoch_seq)
    @property
    def model_bits(self):
        return self.model_bytes * 8

    def get_model_params(self):
        state_dict = self.model.state_dict()
        return state_dict

    def set_model_params(self, model_params_dict: dict):
        state_dict = self.model.state_dict()
        for key, value in state_dict.items():
            state_dict[key] = model_params_dict[key]
        self.model.load_state_dict(state_dict)

    def load_model_params(self, file):
        model_params_dict = get_state_dict(file)
        self.set_model_params(model_params_dict)

    def get_flat_model_params(self):
        flat_params = get_flat_params_from(self.model)
        return flat_params.detach()

    def set_flat_model_params(self, flat_params):
        set_flat_params_to(self.model, flat_params)

    def get_flat_grads(self, dataloader):
        self.optimizer.zero_grad()
        loss, total_num = 0., 0
        for x, y in dataloader:
            if self.gpu:
                x, y = x.cuda(), y.cuda()
            pred = self.model(x)
            if type(pred) == tuple:
                pred = pred[0]
            loss += criterion(pred, y) * y.size(0)
            total_num += y.size(0)
        loss /= total_num

        flat_grads = get_flat_grad(loss, self.model.parameters(), create_graph=True)
        return flat_grads

    def local_train(self, train_dataloader,num_epoch=None, **kwargs):
        """Train model locally and return new parameter and computation cost

        Args:
            train_dataloader: DataLoader class in Pytorch

        Returns
            1. local_solution: updated new parameter
            2. stat: Dict, contain stats
                2.1 comp: total FLOPS, computed by (# epoch) * (# cifar-10-batches-py) * (# one-shot FLOPS)
                2.2 loss
        """
        self.model.train()
        train_loss = train_acc = train_total = 0
        N = round(self.num_epoch * 10 if num_epoch is None else num_epoch * 10)
        for epoch in range(N):
            train_loss = train_acc = train_total = 0
            for batch_idx, (x, y) in enumerate(train_dataloader):
                # from IPython import embed
                # embed()
                if self.gpu:
                    x, y = x.cuda(), y.cuda()

                self.optimizer.zero_grad()
                pred = self.model(x)

                if torch.isnan(pred.max()):
                    from IPython import embed
                    embed()

                loss = criterion(pred, y)
                loss.backward()
                count += 1
                torch.nn.utils.clip_grad_norm(self.model.parameters(), 60)
                self.optimizer.step()

                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum().item()
                target_size = y.size(0)
                unique_elements1, inverse_indices, counts = torch.unique(predicted, return_inverse=True, return_counts=True)
                unique_elements2, inverse_indices, counts = torch.unique(y, return_inverse=True, return_counts=True)
                print(unique_elements1,unique_elements2)
                train_loss += loss.item() * y.size(0)
                train_acc += correct
                train_total += target_size
        local_solution = self.get_flat_model_params()
        param_dict = {"norm": torch.norm(local_solution).item(),
                      "max": local_solution.max().item(),
                      "min": local_solution.min().item()}
        comp = self.num_epoch * train_total * self.flops
        return_dict = {"comp": comp,
                       "loss": train_loss/train_total,
                       "acc": train_acc/train_total}
        return_dict.update(param_dict)
        return local_solution, return_dict

    def local_test(self, test_dataloader):
        self.model.eval()
        test_loss = test_acc = test_total = 0.
        with torch.no_grad():
            for x, y in test_dataloader:

                # print("test")
                # from IPython import embed
                # embed()
                if self.gpu:
                    x, y = x.cuda(), y.cuda()

                pred = self.model(x)
                if type(pred) == tuple:
                    pred = pred[0]
                loss = criterion(pred, y)
                _, predicted = torch.max(pred, 1)
                # unique_elements1, inverse_indices, counts = torch.unique(predicted, return_inverse=True, return_counts=True)
                # unique_elements2, inverse_indices, counts = torch.unique(y, return_inverse=True, return_counts=True)

                #print(unique_elements1, unique_elements2)
                correct = predicted.eq(y).sum()
                test_acc += correct.item()
                test_loss += loss.item() * y.size(0)
                test_total += y.size(0)

        return test_acc, test_loss


class LrdWorker(Worker):
    def __init__(self, model, optimizer, options):
        self.num_epoch = options['num_epoch']
        super(LrdWorker, self).__init__(model, optimizer, options)
    
    def local_train(self, train_dataloader,num_epoch=None,latest_model=None , **kwargs):
        # current_step = kwargs['T']
        self.model.train()
        train_loss = train_acc = train_total = 0
        # num_epoch = self.num_epoch_seq[round_i] if self.ie else self.num_epoch*10
        N = int(self.num_epoch*10 if num_epoch is None else num_epoch*10)
        for i in range(N):
            x, y = next(iter(train_dataloader))
            if self.gpu:
                x, y = x.cuda(), y.cuda()
            self.optimizer.zero_grad()
            pred = self.model(x)
            if type(pred) == tuple:
                pred = pred[0]
            target_one_hot = F.one_hot(y, num_classes=10).float()
            param_shapes = [param.shape for param in self.model.parameters()]
            restored_params = []
            start_idx = 0
            for shape in param_shapes:
                num_elements = shape.numel()  # 获取参数的总元素数量
                param_tensor = latest_model[start_idx:start_idx + num_elements].view(shape)
                restored_params.append(param_tensor)
                start_idx += num_elements
            
            global_model = copy.deepcopy(self.model)
            for param, restored_param in zip(global_model.parameters(), restored_params):
                param.data = restored_param
            proximal_term = 0
            for w, w_t in zip(self.model.parameters(), global_model.parameters()):
                proximal_term += (w - w_t).norm(2)
            loss = criterion(pred, y) + (0 / 2) * proximal_term
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), 60)
            # lr = 100/(400+current_step+i)
            self.optimizer.step()
            
            _, predicted = torch.max(pred, 1)
            correct = predicted.eq(y).sum().item()
            target_size = y.size(0)
            # unique_elements1, inverse_indices, counts = torch.unique(predicted, return_inverse=True, return_counts=True)
            # unique_elements2, inverse_indices, counts = torch.unique(y, return_inverse=True, return_counts=True)
            # print(unique_elements1,unique_elements2)
            train_loss += loss.item() * y.size(0)
            train_acc += correct
            train_total += target_size
            del x, y, pred, target_one_hot, restored_params, global_model, proximal_term, loss, predicted
            torch.cuda.empty_cache()  
        local_solution = self.get_flat_model_params()
        param_dict = {"norm": torch.norm(local_solution).item(),
            "max": local_solution.max().item(),
            "min": local_solution.min().item()}
        comp = self.num_epoch * train_total * self.flops
        return_dict = {"comp": comp,
            "loss": train_loss/train_total,
                "acc": train_acc/train_total}
        return_dict.update(param_dict)
        return local_solution, return_dict


class LrAdjustWorker(Worker):
    def __init__(self, model, optimizer, options):
        self.num_epoch = options['num_epoch']
        super(LrAdjustWorker, self).__init__(model, optimizer, options)
    
    def local_train(self, train_dataloader, num_epoch=None,**kwargs):
        m = kwargs['multiplier']
        current_lr = self.optimizer.get_current_lr()
        self.optimizer.set_lr(current_lr * m)
        
        self.model.train()
        train_loss = train_acc = train_total = 0
        N = int(self.num_epoch*10 if num_epoch is None else num_epoch*10)
        # print("N is ",N)
        for i in range(N):
            x, y = next(iter(train_dataloader))
            
            if self.gpu:
                x, y = x.cuda(), y.cuda()
        
            self.optimizer.zero_grad()
            pred = self.model(x)
            
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), 60)
            # lr = 100/(400+current_step+i)
            self.optimizer.step()
            
            _, predicted = torch.max(pred, 1)
            correct = predicted.eq(y).sum().item()
            target_size = y.size(0)
            
            train_loss += loss.item() * y.size(0)
            train_acc += correct
        train_total += target_size
        
        local_solution = self.get_flat_model_params()
        param_dict = {"norm": torch.norm(local_solution).item(),
            "max": local_solution.max().item(),
            "min": local_solution.min().item()}
        comp = self.num_epoch * train_total * self.flops
        return_dict = {"comp": comp,
            "loss": train_loss/train_total,
                "acc": train_acc/train_total}
        return_dict.update(param_dict)

        self.optimizer.set_lr(current_lr)
        return local_solution, return_dict



class SFAWorker(Worker):
    def __init__(self, model, optimizer, options):
        self.num_epoch = options['num_epoch']

        class DisDataset(data.Dataset):
            def __init__(self):
                super().__init__()
                self.features = []
                self.labels = []

            def reset(self):
                self.features = []
                self.labels = []

            def put_data(self, features, label):
                for feature in features:
                    self.features.extend(feature.detach())
                    self.labels.extend([label for _ in range(len(feature))])

            def __getitem__(self, index):
                return self.features[index], self.labels[index]

            def __len__(self):
                return len(self.labels)
        class Discriminator_map(nn.Module):
            def __init__(self, input_dim, output_dim):
                super(Discriminator_map, self).__init__()
                self.p = nn.Parameter(torch.zeros(input_dim))
                self.mask = torch.sigmoid(10 * self.p)
                self.discriminator = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, 256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, output_dim),
                )

            def forward(self, x):
                self.mask = torch.sigmoid(10 * self.p)
                x = self.mask * x
                x = self.discriminator(x)
                return x
        self.dis = Discriminator_map(1024,10)

        self.data_dis = DisDataset()
        self.move_model_to_gpu(self.dis,options)
        super(SFAWorker, self).__init__(model, optimizer, options)

    def move_model_to_gpu(self,model, options):
        if 'gpu' in options and (options['gpu'] is True):
            device = 0 if 'device' not in options else options['device']
            torch.cuda.set_device(device)
            torch.backends.cudnn.enabled = True
            model.cuda()
            print('>>> Use gpu on device {}'.format(device))
        else:
            print('>>> Don not use gpu')

    def local_train(self, train_dataloader, num_epoch=None, **kwargs):
        m = kwargs['multiplier']
        cid = kwargs['cid']
        current_lr = self.optimizer.get_current_lr()
        self.optimizer.set_lr(current_lr * m)

        self.model.train()
        train_loss = train_acc = train_total = dis_loss = 0
        n = self.num_epoch * 10 if num_epoch is None else int(num_epoch * 10)
        for i in range(n):
            x, y = next(iter(train_dataloader))
            y_dis = torch.full_like(y,cid)
            if self.gpu:
                x, y, y_dis = x.cuda(), y.cuda(), y_dis.cuda()

            self.optimizer.zero_grad()
            pred, reversed_features = self.model(x,self.dis.mask)
            pred_dis = self.dis(reversed_features)

            dis_loss = criterion(pred_dis, y_dis)
            loss = criterion(pred, y) # + dis_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), 60)
            # lr = 100/(400+current_step+i)
            self.optimizer.step()

            _, predicted = torch.max(pred, 1)
            correct = predicted.eq(y).sum().item()
            target_size = y.size(0)

            train_loss += loss.item() * y.size(0)
            train_acc += correct
            # print("dis_loss :{%.5f}" % (dis_loss))
        train_total += target_size

        local_solution = self.get_flat_model_params()
        param_dict = {"norm": torch.norm(local_solution).item(),
                      "max": local_solution.max().item(),
                      "min": local_solution.min().item()}
        comp = self.num_epoch * train_total * self.flops
        return_dict = {"comp": comp,
                       "loss": train_loss / train_total,
                       "acc": train_acc / train_total}
        return_dict.update(param_dict)

        self.optimizer.set_lr(current_lr)

        return local_solution, return_dict

    def local_test(self, test_dataloader):
        self.model.eval()
        test_loss = test_acc = test_total = 0.
        with torch.no_grad():
            for x, y in test_dataloader:
                # print("test")
                # from IPython import embed
                # embed()
                if self.gpu:
                    x, y = x.cuda(), y.cuda()

                pred,out_dis = self.model(x,self.dis.mask)

                loss = criterion(pred, y)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum()
                print(predicted,y)
                test_acc += correct.item()
                test_loss += loss.item() * y.size(0)
                test_total += y.size(0)

        return test_acc, test_loss

    def test_dis(self):
        train_dataloader = torch.utils.data.DataLoader(dataset=self.data_dis, batch_size=64, shuffle=False, drop_last=False)
        self.dis.eval()

        total = 0
        correct = 0

        for idx, data in enumerate(train_dataloader):
            features, labels = data

            if self.gpu:
                features = features.cuda()
                labels = labels.cuda()

            pre = self.dis(features)

            # print(labels,pre)
            _, pred_label = torch.max(pre.softmax(dim=-1), 1)
            total += labels.size()[0]
            correct += (pred_label == labels).sum().item()
        # print("dis_auc", sum(auc_collector) / len(auc_collector))
        print("dis_avg_acc", correct / float(total), )
        return correct / float(total)

    def train_dis(self, clients,latest_model):
        self.set_flat_model_params(latest_model)
        self.data_dis.reset()
        for c in clients:
            for i in range(5):
                x, y = next(iter(c.train_dataloader))
                if self.gpu:
                    x, y = x.cuda(), y.cuda()
                pred, reversed_features = self.model(x,self.dis.mask)
                self.data_dis.put_data([reversed_features],c.cid)

        acc = self.test_dis()
        print("before train dis_acc", acc)
        train_dataloader = torch.utils.data.DataLoader(dataset=self.data_dis, batch_size=64, shuffle=True, drop_last=False)
        optimizer = torch.optim.Adam(self.dis.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        self.dis.train()

        epoch_loss_collector = []
        auc_collector = []
        total = 0
        correct = 0
        for i in range(10):
            for idx, data in enumerate(train_dataloader):
                features, labels = data

                if self.gpu:
                    features = features.cuda()
                    labels = labels.cuda()

                pre = self.dis(features)
                loss = loss_fn(pre, labels)
                epoch_loss_collector.append(loss.item())
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

                # print(labels,pre)
                _, pred_label = torch.max(pre.softmax(dim=-1), 1)
                total += labels.size()[0]
                correct += (pred_label == labels).sum().item()

            print("dis_loss:{:.5f} dis_acc{:.5f}".format(sum(epoch_loss_collector) / len(epoch_loss_collector),
                                                         correct / float(total)))
        # print("dis_auc", sum(auc_collector) / len(auc_collector))

        print("dis_avg_acc", correct / float(total), )
        print(self.dis.mask)
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)

        self.data_dis.reset()
        return epoch_loss
