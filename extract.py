import argparse
import ast
import datetime
import json
import os
import random
import sys
import time

import numpy as np
import tensorboard_logger
import tqdm

import torch
import torch.nn

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import models

from lib.crnn import CRNN2D_elu2
from lib.dataset import DefaultSet
from lib.utils import AverageMeter


def parse_options():
    parser = argparse.ArgumentParser()

    # load data
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=36, help='Number of workers to load data')

    # resume path
    parser.add_argument('--model_path', type=str, required=True, help='Path to load the saved model')

    # dataset split
    parser.add_argument('--subset', type=str, default='all', help='Dataset subset name for training')

    # specify folder
    parser.add_argument('--data_path', type=str, default='', help='Path to load data')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save results')

    # misc
    parser.add_argument('--seed', type=int, default=227)

    return parser.parse_args()


def inference(dataloader, encoder, softmax, opts):
    encoder.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = torch.zeros(len(dataloader.dataset), opts.feat_dim)
    features = features.cuda()

    end = time.time()
    for batch_index, (indices, inputs, labels) in enumerate(dataloader):
        data_time.update(time.time() - end)

        batch_size = inputs.size(0)
        inputs = inputs.float().cuda()

        # ===================forward=====================
        with torch.no_grad():
            features[indices] = encoder(inputs)

        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        print('\033[F\033[KVal:', end='\t')
        print('[{0}/{1}]'.format(batch_index + 1, len(dataloader)), end='\t')
        print(f'BT {batch_time.val:.3f} ({batch_time.avg:.3f})', end='\t')
        print(f'DT {data_time.val:.3f} ({data_time.avg:.3f})', flush=True)

    return features.cpu()


def main(opts):
    if not torch.cuda.is_available():
        print('Only support GPU mode')
        sys.exit(1)

    # fix all parameters for reproducibility
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    os.environ['PYTHONHASHSEED'] = str(opts.seed)

    torch.manual_seed(opts.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opts.seed)
        torch.cuda.manual_seed_all(opts.seed)

    ########### encoder ##############
    checkpoint = torch.load(opts.model_path)

    opts.feat_dim = checkpoint['opts'].feat_dim
    encoder = CRNN2D_elu2(input_size=1 + checkpoint['opts'].n_fft // 2, feat_dim=checkpoint['opts'].feat_dim, dropout=0)

    state_dict = {key.partition('module.')[2]: checkpoint['model'][key] for key in checkpoint['model'].keys()}
    encoder.load_state_dict(state_dict, strict=True)
    for param in encoder.parameters():
        param.requires_grad = False

    # Multi-GPU
    encoder = encoder.cuda()
    encoder = torch.nn.DataParallel(encoder)
    softmax = torch.nn.Softmax(dim=1).cuda()

    ########### data ##############
    dataset = DefaultSet(opts.data_path, opts.subset, checkpoint['opts'].input_len, checkpoint['opts'].n_fft)
    dataloader = DataLoader(dataset, batch_size=opts.batch_size, shuffle=False,
                            num_workers=opts.num_workers, pin_memory=True, drop_last=False)

    ########### inference ##############
    features = inference(dataloader, encoder, softmax, opts)
    np.savez(opts.save_path, files=dataset.files, features=features.numpy())


if __name__ == '__main__':
    opts = parse_options()
    print(vars(opts))

    main(opts)
