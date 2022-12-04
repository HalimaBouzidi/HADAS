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

from .data.data_loader import build_data_loader

from utils.config import setup
import utils.saver as saver
from utils.progress import AverageMeter, ProgressMeter, accuracy_exits
import utils.logging as logging
from .solver import build_optimizer, build_lr_scheduler
import utils.loss_exit as loss_ops 
from copy import deepcopy
import numpy as np

import utils.comm as comm

def train(subnets_to_be_trained,
        train_loader, 
        val_loader, 
        model, 
        args, 
):

    supernet = model.module \
        if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.batch_size = args.batch_size_per_gpu
    args.lr_scheduler.base_lr = args.lr_scheduler.base_lr * (max(1, args.batch_size // 256))

    torch.cuda.set_device(args.gpu)
    results = []

    for idx, net_id in enumerate(subnets_to_be_trained, start=0):

        supernet.set_active_subnet(
            subnets_to_be_trained[net_id]['resolution'],
            subnets_to_be_trained[net_id]['width'],
            subnets_to_be_trained[net_id]['depth'],
            subnets_to_be_trained[net_id]['kernel_size'],
            subnets_to_be_trained[net_id]['expand_ratio'],
        )

        n_exits = 0
        for i in subnets_to_be_trained[net_id]['depth']:
            n_exits += i
        block_ee = [i for i in range(5, n_exits)]
        n_exits = n_exits - 5
        subnet = supernet.get_active_eex_subnet(block_ee=block_ee, num_ee=len(block_ee), threshold_ee=.5)
        subnet_cfg = supernet.get_active_subnet_settings()

        # Freeze the subnet parameters except the exit blocks   
        for _, param in subnet.named_parameters():
            param.requires_grad = False

        for name, module in subnet.named_modules():
            if(name == 'exits'):
                for subname, param in module.named_parameters():
                    param.requires_grad = True

        # Load the subnet into the gpu
        subnet.cuda(args.gpu)

        # use sync batchnorm
        if getattr(args, 'sync_bn', False):
            subnet.apply(lambda m: setattr(m, 'need_sync', True))

        criterion = loss_ops.CumulativeKLDivergenceExits().cuda(args.gpu)

        args.n_iters_per_epoch = len(train_loader)
        optimizer = build_optimizer(args, subnet)
        lr_scheduler = build_lr_scheduler(args, optimizer)
    
        for epoch in range(args.start_epoch, args.epochs):
            args.curr_epoch = epoch
            train_epoch(epoch, subnet, train_loader, optimizer, criterion, n_exits, args, lr_scheduler=lr_scheduler)

        # Run the final evaluation on the validation set
        with torch.no_grad():
            subnet.eval()
            subnet.reset_running_stats_for_calibration()

            # estimate running mean and running statistics
            for batch_idx, (images, _, _) in enumerate(train_loader):
                if batch_idx >= args.post_bn_calibration_batch_num:
                    break
                if getattr(args, 'use_clean_images_for_subnet_training', False):
                    _, images = images
                images = images.cuda(args.gpu, non_blocking=True)
                subnet(images)  #forward only

        torch.cuda.empty_cache()

        #evaluation
        subnet.eval() # freeze again all running stats
        top1 = [0 for i in range(n_exits+1)]
    
        for batch_idx, (images, target, path) in enumerate(val_loader):     

            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            val_len = len(val_loader)

            # compute output
            output, conf, exits = subnet(images)

            # measure accuracy 
            acc, correct = accuracy_exits(output, target, topk=(1, 5), n_exits=exits, per_sample=True)
                    
            for i, elem in enumerate(acc, start=0):
                top1[i] += elem[0].item()

        torch.cuda.empty_cache()
                
        avg_top1 = [elem/val_len for elem in top1]

        summary = str({
        'net_id': net_id,
        'mode': 'evaluate',
        'epoch': getattr(args, 'curr_epoch', -1),
        'acc1': avg_top1,
        **subnet_cfg
        })

        if args.distributed and getattr(args, 'distributed_val', True):
            results += [summary]
        else:
            group = comm.reduce_eval_results(summary, args.gpu)
            results += group

    return results


def train_epoch(
    epoch, 
    subnet, 
    train_loader, 
    optimizer, 
    criterion, 
    n_exits,
    args, 
    lr_scheduler=None, 
):
    subnet.train()
    num_updates = epoch * len(train_loader)

    for batch_idx, (images, target, _) in enumerate(train_loader):

        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        optimizer.zero_grad()
        output, conf, exits = subnet(images)   
        loss = criterion(output, exits, target, temperature=1) # for KL-divergenece
        loss.backward()
        torch.nn.utils.clip_grad_norm_(subnet.parameters(), 1.0)
        optimizer.step()
        num_updates += 1
        if lr_scheduler is not None:
            lr_scheduler.step()
    
    
    