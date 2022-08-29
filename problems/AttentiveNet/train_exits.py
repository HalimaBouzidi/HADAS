# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import builtins
import math
import os
import random
import shutil
import warnings
import sys
import operator

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from data.data_loader import build_data_loader

from utils.config import setup
import utils.saver as saver
from utils.progress import AverageMeter, ProgressMeter, accuracy_exits
import utils.logging as logging
from evaluate import exit_eval as exit_eval
from solver import build_optimizer, build_lr_scheduler
import utils.loss_exit as loss_ops 
import models
from copy import deepcopy
import numpy as np
import joblib 

parser = argparse.ArgumentParser(description='Exit-blocks Training')
parser.add_argument('--config-file', default=None, type=str, help='training configuration')
parser.add_argument('--model', default='a0', type=str, help='model to evaluate')
parser.add_argument('--gpu', default=0, type=int, help='gpu local rank')
logger = logging.get_logger(__name__)

def build_args_and_env(run_args):

    assert run_args.config_file and os.path.isfile(run_args.config_file), 'cannot locate config file'
    args = setup(run_args.config_file)
    args.model = run_args.model
    args.gpu = run_args.gpu
    args.config_file = run_args.config_file
    args.exp_name = 'attentive_nas_model_'+args.model+'_'+args.dataset+'_exits'
    args.models_save_dir = os.path.join(args.models_save_dir, args.exp_name)

    if not os.path.exists(args.models_save_dir):
        os.makedirs(args.models_save_dir)

    #backup config file
    saver.copy_file(args.config_file, '{}/{}'.format(args.models_save_dir, os.path.basename(args.config_file)))

    args.checkpoint_save_path = os.path.join(
        args.models_save_dir, 'attentive_nas.pth.tar'
    )
    args.logging_save_path = os.path.join(
        args.models_save_dir, f'stdout.log'
    )
    return args


def main():
    run_args = parser.parse_args()
    args = build_args_and_env(run_args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.batch_size = args.batch_size_per_gpu
    args.lr_scheduler.base_lr = args.lr_scheduler.base_lr * (max(1, args.batch_size // 256))

    # set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu:
        torch.cuda.manual_seed(args.seed)

    # Setup logging format.
    logging.setup_logging(args.logging_save_path, 'w')
    torch.cuda.set_device(args.gpu)

    # build model
    logger.info("=> creating model '{}'".format(args.arch))
    args.__dict__['active_subnet'] = args.__dict__['pareto_models'][args.model]
    n_exits = args.__dict__['active_subnet']['num_ee'] 
    model = models.model_factory.create_model(args)

    # Freeze the model parameters except the exit blocks   
    for _, param in model.named_parameters():
        param.requires_grad = False

    for name, module in model.named_modules():
        if(name == 'exits'):
            for subname, param in module.named_parameters():
                param.requires_grad = True

    # Load the model into the gpu
    model.cuda(args.gpu)

    # use sync batchnorm
    if getattr(args, 'sync_bn', False):
        model.apply(lambda m: setattr(m, 'need_sync', True))

    logger.info(model)

    if args.loss == 'kl-div':
        criterion = loss_ops.CumulativeKLDivergenceExits().cuda(args.gpu)
    elif args.loss == 'nll':
        criterion = loss_ops.CumulativeNLLExits().cuda(args.gpu)
    else:
        raise NotImplementedError

    ## load dataset, train_sampler
    logger.info("=> loading the training dataset '{}'".format(args.dataset))
    train_loader, val_loader, train_sampler =  build_data_loader(args)
    args.n_iters_per_epoch = len(train_loader)

    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)
 
    # optionally resume from a checkpoint
    if args.resume:
        logger.info("Resuming from checkpoint...")
        saver.load_checkpoints(args, model, optimizer, lr_scheduler, logger)

    logger.info(args)

    for epoch in range(args.start_epoch, args.epochs):

        args.curr_epoch = epoch
        logger.info('Training lr {}'.format(lr_scheduler.get_lr()[0]))

        # train for one epoch
        acc1, acc5 = train_epoch(epoch, model, train_loader, optimizer, criterion, n_exits, args, lr_scheduler=lr_scheduler)

        if (epoch % 5 == 0 or epoch == args.epochs-1): # validate every 5 epochs
            # Run the evaluation on the validation set
            validate(train_loader, val_loader, model, criterion, n_exits, args)

        # save checkpoints
        saver.save_checkpoint(
            args.checkpoint_save_path, 
            model,
            optimizer,
            lr_scheduler, 
            args,
            epoch,
        )

def train_epoch(
    epoch, 
    model, 
    train_loader, 
    optimizer, 
    criterion, 
    n_exits,
    args, 
    lr_scheduler=None, 
):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    num_updates = epoch * len(train_loader)

    for batch_idx, (images, target) in enumerate(train_loader):

        losses_l = [0 for i in range(n_exits+1)]
        top1_l = [0 for i in range(n_exits+1)]
        top5_l = [0 for i in range(n_exits+1)]

        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        optimizer.zero_grad()

        output, conf, exits = model(images)
        
        if args.loss == 'kl-div':
            loss = criterion(output, exits, target, temperature=1.) # for KL-divergenece
        else:
            loss = criterion(output, exits, target) #n for negative log likelihood loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        #accuracy measured on the current batch
        acc = accuracy_exits(output, target, topk=(1, 5), n_exits=exits, per_sample=False)
        
        for i, elem in enumerate(acc, start=0):
            top1_l[i] += elem[0].item()
            top5_l[i] += elem[1].item()
        
        for i in range(exits+1):
            losses_l[i] += loss.item()

        losses.update(np.mean(losses_l), images.size(0))
        top1.update(np.mean(top1_l), images.size(0))
        top5.update(np.mean(top5_l), images.size(0))

        num_updates += 1
        if lr_scheduler is not None:
            lr_scheduler.step()

        if batch_idx % args.print_freq == 0:
            progress.display(batch_idx, logger)

    return top1.avg, top5.avg
    

def validate(
    train_loader, 
    val_loader, 
    model, 
    criterion, 
    n_exits,
    args, 
    distributed = False,
):

    criterion = nn.CrossEntropyLoss().cuda()
    exit_eval.validate_eex(
        train_loader,
        val_loader, 
        model, 
        criterion,
        n_exits,
        args,
        logger,
        bn_calibration = True,
    )


if __name__ == '__main__':
    main()

