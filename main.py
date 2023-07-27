import argparse
import os
import random
import sys
from datetime import datetime

import numpy
import torch
from dateutil import tz

import config
from config import TASKS, PERSONALISATION, HUMOR, MIMIC, AROUSAL, VALENCE, PERSONALISATION_DIMS, MODELS
from data_parser import load_data
from dataset import MuSeDataset, custom_collate_fn
from eval import evaluate, calc_ccc, calc_auc, mean_pearsons
from loss import CCCLoss, BCELossWrapper, MSELossWrapper
from model import Model, TFModel, TFT_encoder
from train import train_model
from utils import Logger, seed_worker, log_results
import wandb


def parse_args():  # done
    parser = argparse.ArgumentParser(description='MuSe 2023.')

    parser.add_argument('--model', type=str, required=True, choices=MODELS,
                        help=f'Specify the model type from {MODELS}.')
    parser.add_argument('--project', type=str, default='test',
                        help='wandb project name')
    parser.add_argument('--feature', required=True,
                        help='Specify the features used (only one).')
    parser.add_argument('--emo_dim', default=AROUSAL, choices=PERSONALISATION_DIMS,
                        help=f'Specify the emotion dimension, only relevant for personalisation (default: {AROUSAL}).')
    parser.add_argument('--normalize', type=int, default=0,
                        help='Specify whether to normalize features (default: False).')
    parser.add_argument('--win_len', type=int, default=200,
                        help='Specify the window length for segmentation (default: 200 frames).')
    parser.add_argument('--hop_len', type=int, default=100,
                        help='Specify the hop length to for segmentation (default: 100 frames).')
    parser.add_argument('--model_dim', type=int, default=64,
                        help='Specify the number of hidden states in the RNN (default: 64).')
    parser.add_argument('--rnn_n_layers', type=int, default=1,
                        help='Specify the number of layers for the RNN (default: 1).')
    parser.add_argument('--rnn_bi', action='store_true',
                        help='Specify whether the RNN is bidirectional or not (default: False).')
    parser.add_argument('--d_fc_out', type=int, default=64,
                        help='Specify the number of hidden neurons in the output layer (default: 64).')
    parser.add_argument('--rnn_dropout', type=float, default=0.2)
    parser.add_argument('--linear_dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=100,
                        help='Specify the number of epochs (default: 100).')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Specify the batch size (default: 256).')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Specify initial learning rate (default: 0.0001).')
    parser.add_argument('--seed', type=int, default=101,
                        help='Specify the initial random seed (default: 101).')
    parser.add_argument('--result_csv', default=None, help='Append the results to this csv (or create it, if it '
                                                           'does not exist yet). Incompatible with --predict')
    parser.add_argument('--early_stopping_patience', type=int, default=15, help='Patience for early stopping')
    parser.add_argument('--reduce_lr_patience', type=int, default=5, help='Patience for reduction of learning rate')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Specify whether to use gpu for training (default: False).')
    parser.add_argument('--cache', action='store_true',
                        help='Specify whether to cache data as pickle file (default: False).')
    parser.add_argument('--regularization', type=float, required=False, default=0.0,
                        help='L2-Penalty')
    # evaluation only arguments

    args = parser.parse_args()
    args.normalize = True if args.normalize == 1 else False
    if not (args.result_csv is None) and args.predict:
        print("--result_csv is not compatible with --predict")
        sys.exit(-1)
    return args


def get_loss_fn(task):  # done
    if task == PERSONALISATION:
        return CCCLoss(), 'CCC'
    elif task == HUMOR:
        return BCELossWrapper(), 'Binary Crossentropy'
    elif task == MIMIC:
        return MSELossWrapper(reduction='mean'), 'MSE'


def get_eval_fn(task):  # done
    if task == PERSONALISATION:
        return calc_ccc, 'CCC'
    elif task == MIMIC:
        return mean_pearsons, 'Mean Pearsons'
    elif task == HUMOR:
        return calc_auc, 'AUC-Score'


def main(args):
    # ensure reproducibility
    numpy.random.seed(10)  # done by pl.seed everything
    random.seed(10)  # done
    torch.manual_seed(args.seed)  # done
    model_dict = {"RNN": Model, "TF": TFModel, "TFT": TFT_encoder}

    # emo_dim only relevant for stress/personalisation
    print('Loading data ...')
    data = load_data(args.task, args.paths, args.feature, args.emo_dim, args.normalize,
                     args.win_len, args.hop_len, save=args.cache)  # done
    datasets = {partition: MuSeDataset(data, partition) for partition in data.keys()}

    args.d_in = datasets['train'].get_feature_dim()  # args.d_in = self.data_loader['train'].dataset.get_feature_dim()

    args.n_targets = config.NUM_TARGETS[args.task]  # done
    args.n_to_1 = args.task in config.N_TO_1_TASKS  # lightning datamodule prepare_data
    loss_fn, loss_str = get_loss_fn(args.task)  # basemodel from here
    eval_fn, eval_str = get_eval_fn(args.task)

    wandb.init(project=args.project)
    wandb.run.name = args.log_file_name + f"-{args.seed}"
    # move data initialisation below here...
    torch.manual_seed(args.seed)
    data_loader = {}
    for partition, dataset in datasets.items():  # one DataLoader for each partition
        batch_size = args.batch_size if partition == 'train' else (
            1 if args.task == PERSONALISATION else 2 * args.batch_size)
        shuffle = True if partition == 'train' else False  # shuffle only for train partition
        data_loader[partition] = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                                             num_workers=4,
                                                             worker_init_fn=seed_worker,
                                                             collate_fn=custom_collate_fn)

    model = model_dict[args.model](args)

    print('=' * 50)
    print(f'Training model... [seed {args.seed}] for at most {args.epochs} epochs')

    val_loss, val_score, best_model_file = train_model(args.task, model, data_loader, args.epochs,
                                                       args.lr, args.paths['model'], args.seed,
                                                       use_gpu=args.use_gpu,
                                                       loss_fn=loss_fn, eval_fn=eval_fn,
                                                       eval_metric_str=eval_str,
                                                       regularization=args.regularization,
                                                       early_stopping_patience=args.early_stopping_patience,
                                                       reduce_lr_patience=args.reduce_lr_patience)

    wandb.finish()


if __name__ == '__main__':
    args = parse_args()
    args.use_gpu = True

    args.log_file_name = '{}_{}_[{}]_[{}]_[{}_{}_{}_{}]_[{}_{}]_{}'.format(args.model,
                                                                           datetime.now(tz=tz.gettz()).strftime(
                                                                               "%Y-%m-%d-%H-%M"), args.feature,
                                                                           args.emo_dim,
                                                                           args.model_dim, args.rnn_n_layers,
                                                                           args.rnn_bi,
                                                                           args.d_fc_out, args.lr,
                                                                           args.batch_size, args.win_len)

    # adjust your paths in config.py
    args.task = "personalisation"
    args.paths = {
        'log': os.path.join(config.LOG_FOLDER, args.task),
        'data': os.path.join(config.DATA_FOLDER, args.task),
        'model': os.path.join(config.MODEL_FOLDER, args.task,
                              args.log_file_name)}

    for folder in args.paths.values():
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

    args.paths.update({'features': config.PATH_TO_FEATURES[args.task],
                       'labels': config.PATH_TO_LABELS[args.task],
                       'partition': config.PARTITION_FILES[args.task]})

    sys.stdout = Logger(os.path.join(args.paths['log'], args.log_file_name + '.txt'))  # Not_implemented, ask about this
    print(' '.join(sys.argv))
    main(args)
