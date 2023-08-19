import torch
import argparse
import shutil
import os, sys
from pathlib import Path

if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')
from utils.learning.train_part import train

if os.getcwd() + '/utils/common/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/common/')
from utils.common.utils import seed_fix

from create_h5 import write_h5

def parse():
    parser = argparse.ArgumentParser(description='Train Varnet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-e', '--num-epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('-l', '--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-r', '--report-interval', type=int, default=500, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='e2evarnet', help='Name of network')
    parser.add_argument('-t', '--data-path', type=Path, default='/Data/', help='Directory of data')
    parser.add_argument('-o', '--out-path', type=Path, default='../recon_data/recon_varnet/')
    
    parser.add_argument('--cascade', type=int, default=2, help='Number of cascades | Should be less than 12') ## important hyperparameter
    parser.add_argument('--chans', type=int, default=18, help='Number of channels for cascade U-Net | 18 in original varnet') ## important hyperparameter
    parser.add_argument('--sens_chans', type=int, default=8, help='Number of channels for sensitivity map U-Net | 8 in original varnet') ## important hyperparameter
    parser.add_argument('--input-key', type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--seed', type=int, default=430, help='Fix random seed')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    
    # fix seed
    if args.seed is not None:
        seed_fix(args.seed)
    
    args.exp_dir = '../result' / args.net_name / 'checkpoints'
    
    args.data_path_train = args.data_path / 'train'
    args.data_path_val = args.data_path / 'val'
    args.data_path_leaderboard_acc4 = args.data_path / 'leaderboard' / 'acc4'
    args.data_path_leaderboard_acc4 = args.data_path / 'leaderboard' / 'acc8'
    
    args.out_path_train = args.out_path / 'train'
    args.out_path_val = args.out_path / 'val'
    args.out_path_leaderboard_acc4 = args.out_path / 'acc4'
    args.out_path_leaderboard_acc8 = args.out_path / 'acc8'
    
    args.out_path_train.mkdir(parents=True, exist_ok=True)
    args.out_path_val.mkdir(parents=True, exist_ok=True)
    args.out_path_leaderboard_acc4.mkdir(parents=True, exist_ok=True)
    args.out_path_leaderboard_acc8.mkdir(parents=True, exist_ok=True)

    write_h5(args)