# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from .utils.progress import AverageMeter, accuracy, entropy

def log_helper(summary, logger=None):
    if logger:
        logger.info(summary)
    else:
        print(summary)


def validate_one_subnet(
    val_loader,
    subnet,
    args, 
    logger=None, 
):
    top1 = AverageMeter('Acc@1', ':6.2f')

    subnet.cuda(args.gpu)
    subnet.eval() # freeze again all running stats
   
    for batch_idx, (images, target, _) in enumerate(val_loader):     
    # path for CIFAR datasets is the index

        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = subnet(images)

        # measure accuracy 
        (acc1, acc5), correct = accuracy(output, target, topk=(1, 5), per_sample=True)
        
        batch_size = images.size(0)
        top1.update(acc1, batch_size)

    return float(top1.avg)