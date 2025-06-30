import numpy as np
import argparse
import importlib
import torch
import os
import random
from src.utils.worker_utils import read_data
from config import OPTIMIZERS, DATASETS, MODEL_PARAMS, TRAINERS
import gc

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def read_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('--algo',
                        help='name of trainer;',
                        type=str,
                        choices=OPTIMIZERS,
                        default='fedavg4')
    parser.add_argument('--optim',
                        help='optimizer;',
                        type=str,
                        default='GD')
    parser.add_argument('--dataset',
                        help='name of dataset;',
                        type=str,
                        default='mnist_all_data_1_random_niid')
    parser.add_argument('--model',
                        help='name of model;',
                        type=str,
                        default='cnn')
    parser.add_argument('--wd',
                        help='weight decay parameter;',
                        type=float,
                        default=0.001)
    parser.add_argument('--gpu',
                        type=str2bool,
                        default=False,
                        help='use gpu (default: False)')
    parser.add_argument('--noprint',
                        action='store_true',
                        default=False,
                        help='whether to print inner result (default: False)')
    parser.add_argument('--noaverage',
                        action='store_true',
                        default=False,
                        help='whether to only average local solutions (default: True)')
    parser.add_argument('--device',
                        help='selected CUDA device',
                        default=0,
                        type=int)
    parser.add_argument('--num_round',
                        help='number of rounds to simulate;',
                        type=int,
                        default=10)
    parser.add_argument('--eval_every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=5)
    parser.add_argument('--clients_per_round',
                        help='number of clients trained per round;',
                        type=int,
                        default=10)
    parser.add_argument('--batch_size',
                        help='batch size when clients train on cifar-10-batches-py;',
                        type=int,
                        default=64)
    parser.add_argument('--num_epoch',
                        help='number of epochs when clients train on cifar-10-batches-py;',
                        type=int,
                        default=5)
    parser.add_argument('--lr',
                        help='learning rate for inner solver;',
                        type=float,
                        default=0.1)
    parser.add_argument('--seed',
                        help='seed for randomness;',
                        type=int,
                        default=0)
    parser.add_argument('--real_gradient',
                        help='use real gradient to sample;',
                        type=str2bool,
                        default=False)
    parser.add_argument('--projection',
                        help='project raw gradient to low dim;',
                        type=str2bool,
                        default=False)
    parser.add_argument('--compress_dim',
                        help='compress dim of gradient;',
                        type=int,
                        default=5000)
    parser.add_argument('--decrease_E',
                        help='use decrease E;',
                        type=str2bool,
                        default=False)
    parser.add_argument('--repeat',
                        help='use different seed to repeat experience;',
                        type=str2bool,
                        default=False)
    parser.add_argument('--dynamic_E',
                        help='dynamic adjust E by distance;',
                        type=str2bool,
                        default=False)
    parser.add_argument('--del_client',
                        help='del client;',
                        type=str2bool,
                        default=False)
    parser.add_argument('--momentum',
                        help='use momentum gradient;',
                        type=str2bool,
                        default=False)
    parser.add_argument('--random_gradient',
                        help='use random gradient;',
                        type=str2bool,
                        default=False)
    parser.add_argument('--min_similarity',
                        help='control min similarity;',
                        type=float,
                        default=0.2)
    parser.add_argument('--dis',
                        help='add more information;',
                        type=str,
                        default='')
    parser.add_argument('--note',
                        help='more information',
                        type=str,
                        default='ori')
    parser.add_argument('--preamble',
                        help='preamble;',
                        type=int,
                        default=10)
    parsed = parser.parse_args()
    options = parsed.__dict__
    print(options['gpu'], torch.cuda.is_available())
    options['gpu'] = options['gpu'] and torch.cuda.is_available()
    # Set seeds
    random.seed(options['seed'])
    np.random.seed(1 + options['seed'])
    torch.manual_seed(12 + options['seed'])
    if options['gpu']:
        torch.cuda.manual_seed_all(123 + options['seed'])

    # read cifar-10-batches-py
    idx = options['dataset'].find("_")
    if idx != -1:
        dataset_name, sub_data = options['dataset'][:idx], options['dataset'][idx+1:]
    else:
        dataset_name, sub_data = options['dataset'], None
    assert dataset_name in DATASETS, "{} not in dataset {}!".format(dataset_name, DATASETS)

    # Add model arguments
    options.update(MODEL_PARAMS(dataset_name, options['model']))

    # Load selected trainer
    # different algo like uniform or md
    trainer_path = 'src.trainers.%s' % options['algo']
    mod = importlib.import_module(trainer_path)
    trainer_class = getattr(mod, TRAINERS[options['algo']])

    # Print arguments and return
    max_length = max([len(key) for key in options.keys()])
    fmt_string = '\t%' + str(max_length) + 's : %s'
    print('>>> Arguments:')
    for keyPair in sorted(options.items()):
        print(fmt_string % keyPair)
    return options, trainer_class, dataset_name, sub_data


def main():
    # Parse command line arguments
    options, trainer_class, dataset_name, sub_data = read_options()
    train_path = os.path.join('./data', dataset_name, 'data', 'train')
    test_path = os.path.join('./data', dataset_name, 'data', 'test')
    
    # `dataset` is a tuple like (cids, groups, train_data, test_data)
    all_data_info = read_data(train_path, test_path, sub_data)
    cids, groups, train_data, test_data = all_data_info

    accs = []
    if options['repeat']:
        # repeat experiment in three times in different seed
        for i in range(3):
            options['seed'] = i
            print("run in seed=",options['seed'])
            trainer = trainer_class(options, all_data_info)
            trainer.train(options)
            acc = trainer.max_test_acc[0]
            accs.append(acc)
            del trainer  
            gc.collect()  
            torch.cuda.empty_cache()
        print(accs,'average',np.average(accs))
    else:
        print("run in seed=", options['seed'])
        trainer = trainer_class(options, all_data_info)
        trainer.train()

    max_length = max([len(key) for key in options.keys()])
    fmt_string = '\t%' + str(max_length) + 's : %s'
    print('>>> Arguments:')
    for keyPair in sorted(options.items()):
        print(fmt_string % keyPair)

if __name__ == '__main__':
    main()
