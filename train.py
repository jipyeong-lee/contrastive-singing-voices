import argparse
import ast
import datetime
import json
import os
import sys
import random
import sys
import time

import numpy as np
import tensorboard_logger

import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import torchvision
from torchvision import models

from lib.crnn import CRNN2D_elu2
from lib.dataset import ContrastiveSet
from lib.nce_average import NCEAverage, NCEAverageNeg
from lib.nce_criterion import NCECriterion, NCESoftmaxLoss
from lib.utils import AverageMeter


def parse_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=150, help='Train a model for this epochs')
    parser.add_argument('--print_freq', type=int, default=10, help='Print training losses for each this batchs')
    parser.add_argument('--save_freq', type=int, default=5, help='Save intermediate models for each this epochs')

    # load data
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=35, help='Number of workers to load data')
    parser.add_argument('--n_fft', type=int, default=800, help='Size of FFT to be applied to the input data')
    parser.add_argument('--input_len', type=int, default=80000, help='Length of the input data for the time axis')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')

    # resume path
    parser.add_argument('--resume', type=str, default='', metavar='PATH', help='Path to the latest checkpoint (default: none)')

    # model definition
    parser.add_argument('--softmax', type=ast.literal_eval, default=True)
    parser.add_argument('--nce_k', type=int, default=4096)
    parser.add_argument('--nce_t', type=float, default=0.07)
    parser.add_argument('--nce_m', type=float, default=0.5)

    # contrastive conditions
    parser.add_argument('--pitch', type=str, required=True, choices=['none', 'pos', 'neg'])
    parser.add_argument('--stretch', type=str, required=True, choices=['none', 'pos', 'neg'])

    # model parameters
    parser.add_argument('--feat_dim', type=int, default=256, help='Dimension of the features used for inner product')
    parser.add_argument('--dropout', type=float, default=0.1, help='Ratio of the dropout for training')

    # specify folder
    parser.add_argument('--data_path', type=str, default='', help='Path to load data')
    parser.add_argument('--save_path', type=str, default='', help='Path to save results')

    # misc
    parser.add_argument('--seed', type=int, default=227)

    opts = parser.parse_args()

    opts.save_path = opts.save_path or os.path.join(os.path.dirname(__file__), 'runs')
    opts.model_name = datetime.datetime.now().strftime('train_%Y%m%d_%H%M%S')

    opts.model_folder = os.path.join(opts.save_path, opts.model_name, 'ckpt')
    if not os.path.isdir(opts.model_folder):
        os.makedirs(opts.model_folder)

    opts.tb_folder = os.path.join(opts.save_path, opts.model_name, 'logs')
    if not os.path.isdir(opts.tb_folder):
        os.makedirs(opts.tb_folder)

    with open(os.path.join(opts.save_path, opts.model_name, 'params.json'), 'w') as fh:
        json.dump(opts.__dict__, fh, indent=4)

    return opts


def set_model(opts, n_data):
    model = CRNN2D_elu2(input_size=1 + opts.n_fft // 2, feat_dim=opts.feat_dim, dropout=opts.dropout)
    if opts.pitch == 'neg' or opts.stretch == 'neg':
        contrast = NCEAverageNeg(opts.feat_dim, n_data, opts.nce_k, opts.nce_t, opts.nce_m, opts.softmax)
    else:
        contrast = NCEAverage(opts.feat_dim, n_data, opts.nce_k, opts.nce_t, opts.nce_m, opts.softmax)

    criterion_1 = NCESoftmaxLoss() if opts.softmax else NCECriterion(n_data)
    criterion_2 = NCESoftmaxLoss() if opts.softmax else NCECriterion(n_data)

    # GPU mode
    model = model.cuda()
    contrast = contrast.cuda()
    criterion_1 = criterion_1.cuda()
    criterion_2 = criterion_2.cuda()

    # Multi-GPU
    model = torch.nn.DataParallel(model)

    return model, contrast, criterion_1, criterion_2


def train(epoch, dataloader, model, contrast, criterion_1, criterion_2, optimizer, opts):
    model.train()
    contrast.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    view1_loss_meter = AverageMeter()
    view2_loss_meter = AverageMeter()
    view1_prob_meter = AverageMeter()
    view2_prob_meter = AverageMeter()

    end = time.time()
    for batch_index, (data_index, inputs_ori, inputs_pos, inputs_neg) in enumerate(dataloader):
        data_time.update(time.time() - end)

        data_index = data_index.cuda()
        batch_size = inputs_ori.size(0)

        inputs_ori = inputs_ori.float().cuda()
        inputs_pos = inputs_pos.float().cuda()
        inputs_neg = inputs_neg.float().cuda()

        # ===================forward=====================
        feat_ori = model(inputs_ori)
        feat_pos = model(inputs_pos)

        if opts.pitch == 'neg' or opts.stretch == 'neg':
            feat_neg = model(inputs_neg)
            out_1, out_2 = contrast(feat_ori, feat_pos, feat_neg, data_index)
        else:
            out_1, out_2 = contrast(feat_ori, feat_pos, data_index)

        view1_loss = criterion_1(out_1)
        view2_loss = criterion_2(out_2)
        view1_prob = out_1[:, 0].mean()
        view2_prob = out_2[:, 0].mean()

        loss = view1_loss + view2_loss

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        losses.update(loss.item(), batch_size)
        view1_loss_meter.update(view1_loss.item(), batch_size)
        view1_prob_meter.update(view1_prob.item(), batch_size)
        view2_loss_meter.update(view2_loss.item(), batch_size)
        view2_prob_meter.update(view2_prob.item(), batch_size)

        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (batch_index + 1) % opts.print_freq == 0:
            print('\033[F\033[K', end='')
            print('Train: [{0}/{1}][{2}/{3}]'.format(epoch, opts.epochs, batch_index + 1, len(dataloader)), end='\t')
            print(f'BT {batch_time.val:.3f} ({batch_time.avg:.3f})', end='\t')
            print(f'DT {data_time.val:.3f} ({data_time.avg:.3f})', end='\t')
            print(f'Loss {losses.val:.3f} ({losses.avg:.3f})', end='\t')
            print('1_p {probs1.val:.3f} ({probs1.avg:.3f})'.format(probs1=view1_prob_meter), end='\t')
            print('2_p {probs2.val:.3f} ({probs2.avg:.3f})'.format(probs2=view2_prob_meter), flush=True)

    return view1_loss_meter.avg, view1_prob_meter.avg, view2_loss_meter.avg, view2_prob_meter.avg


def main(opts):
    if not torch.cuda.is_available():
        print('Only support GPU mode')
        sys.exit(1)

    # fix all parameters for reproducibility
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    os.environ['PYTHONHASHSEED'] = str(opts.seed)

    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)

    # set the data loader
    dataset = ContrastiveSet(opts.data_path, 'train', opts.input_len, opts.n_fft, opts.pitch, opts.stretch)
    dataloader = DataLoader(dataset, batch_size=opts.batch_size, shuffle=True,
                            num_workers=opts.num_workers, pin_memory=True, drop_last=True)

    # set the model
    model, contrast, criterion_1, criterion_2 = set_model(opts, len(dataset))

    # set the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.learning_rate, weight_decay=opts.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50, 75, 100], gamma=0.2)

    # optionally resume from a checkpoint
    opts.start_epoch = 1

    if opts.resume:
        if os.path.isfile(opts.resume):
            print('===> loading checkpoint {}'.format(opts.resume))
            checkpoint = torch.load(opts.resume, map_location='cpu')

            opts.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            contrast.load_state_dict(checkpoint['contrast'])

            print('===> loaded checkpoint {} (epoch {})'.format(opts.resume, checkpoint['epoch']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print('===> no checkpoint found at {}'.format(opts.resume))

    # tensorboard
    logger = tensorboard_logger.Logger(logdir=opts.tb_folder, flush_secs=2)

    # routine
    for epoch in range(opts.start_epoch, opts.epochs + 1):
        time1 = time.time()
        view1_loss, view1_prob, view2_loss, view2_prob = train(epoch, dataloader, model, contrast, criterion_1, criterion_2, optimizer, opts)
        time2 = time.time()
        print('\nepoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('view1_loss', view1_loss, epoch)
        logger.log_value('view1_prob', view1_prob, epoch)
        logger.log_value('view2_loss', view2_loss, epoch)
        logger.log_value('view2_prob', view2_prob, epoch)

        # save model
        if epoch % opts.save_freq == 0:
            state = {
                'opts': opts,
                'model': model.state_dict(),
                'contrast': contrast.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            save_file = os.path.join(opts.model_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))

            print('==> saving to {} ...'.format(save_file))
            torch.save(state, save_file)

            # help release GPU memory
            del state

        torch.cuda.empty_cache()
        scheduler.step()

    print('==> finished training for {}'.format(opts.model_name))


if __name__ == '__main__':
    opts = parse_options()
    print(vars(opts))

    main(opts)
