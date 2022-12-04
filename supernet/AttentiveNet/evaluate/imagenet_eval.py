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

def validate_one_subnet(
    val_loader,
    subnet,
    criterion,
    args, 
    logger=None, 
):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    subnet.cuda(args.gpu)
    subnet.eval() # freeze again all running stats

    path_list = []
    entropy_list = []
    correct_list = []
   
    for batch_idx, (images, target, path) in enumerate(val_loader):     
    # path for CIFAR datasets is the index

        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = subnet(images)
        batch_entropy = entropy(output)

        loss = criterion(output, target).item()

        # measure accuracy 
        (acc1, acc5), correct = accuracy(output, target, topk=(1, 5), per_sample=True)
        
        batch_size = images.size(0)
        
        if args.distributed and getattr(args, 'distributed_val', True):
            corr1, corr5, loss = acc1 * batch_size, acc5 * batch_size, loss * batch_size
            stats = torch.tensor([corr1, corr5, loss, batch_size], device=args.gpu)
            dist.barrier()  # synchronizes all processes
            dist.all_reduce(stats, op=torch.distributed.ReduceOp.SUM) 
            corr1, corr5, loss, batch_size = stats.tolist()
            acc1, acc5, loss = corr1 / batch_size, corr5 / batch_size, loss/batch_size

        top1.update(acc1, batch_size)
        top5.update(acc5, batch_size)
        losses.update(loss, batch_size)

        entropy_list.extend(batch_entropy)
        path_list.extend(path)
        correct_list.extend(correct[0])     # Top 1

    return float(top1.avg), float(top5.avg), float(losses.avg), path_list, entropy_list, correct_list