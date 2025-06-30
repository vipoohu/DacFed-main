# GLOBAL PARAMETERS
DATASETS = ['sent140', 'nist', 'shakespeare','femnist',
            'mnist', 'synthetic', 'cifar10','medmnist']
TRAINERS = {
            'fedavg': 'FedAvg5Trainer',
            'HGA': 'FedAvg6Trainer',
            'AEA': 'FedAvg9Trainer',
            'HGA_AEA': 'FedAvg96Trainer',
            'FCS' : 'FPSTrainer',
            'FCS_HGA' : 'FPSTrainer6',
            'FPSgai' : 'FPSTrainergai',
            'FPSgai9' : 'FPSTrainergai9',
            'DacFed' : 'FPSTrainergai96',
}
OPTIMIZERS = TRAINERS.keys()


class ModelConfig(object):
    def __init__(self):
        pass

    def __call__(self, dataset, model):
        dataset = dataset.split('_')[0]
        if dataset == 'mnist' or dataset == 'nist' or dataset == 'medmnist':
            if model == 'logistic' or model == '2nn':
                return {'input_shape': 784, 'num_class': 10}
            else:
                if dataset == 'medmnist':

                    return {'input_shape': (3, 28, 28), 'num_class':9}
                else:
                    return {'input_shape': (1, 28, 28), 'num_class': 10}
        elif dataset == 'femnist':
            if model == 'logistic' or model == '2nn':
                return {'input_shape': 784, 'num_class': 62}
            else:
                return {'input_shape': (1, 28, 28), 'num_class': 62}
        elif dataset == 'cifar10':
            return {'input_shape': ( 3,32, 32), 'num_class': 10}
        elif dataset == 'sent140':
            sent140 = {'bag_dnn': {'num_class': 2},
                       'stacked_lstm': {'seq_len': 25, 'num_class': 2, 'num_hidden': 100},
                       'stacked_lstm_no_embeddings': {'seq_len': 25, 'num_class': 2, 'num_hidden': 100}
                       }
            return sent140[model]
        elif dataset == 'shakespeare':
            shakespeare = {'stacked_lstm': {'seq_len': 80, 'emb_dim': 80, 'num_hidden': 256}
                           }
            return shakespeare[model]
        elif dataset == 'synthetic':
            return {'input_shape': 60, 'num_class': 10}
        else:
            raise ValueError('Not support dataset {}!'.format(dataset))


MODEL_PARAMS = ModelConfig()
