# Test script to obtain the entropy levels of the different data samples on multiple exits

import argparse
import builtins
import math
import os
import random
import shutil
import time
import sys
import warnings
import argparse
from datetime import date
import csv
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from utils.config import setup
from utils.flops_counter import count_net_flops_and_params
import utils.comm as comm
import utils.saver as saver

import models
from data.data_loader import build_data_loader
from utils.progress import AverageMeter, ProgressMeter, accuracy

parser = argparse.ArgumentParser(description='Test AttentiveNas Models')
parser.add_argument('--config-file', default='./configs/eval_attentive_nas_models.yml')
parser.add_argument('--model', default='a0', type=str, choices=['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6'])
parser.add_argument('--gpu', default=0, type=int, help='gpu id')

run_args = parser.parse_args()

if __name__ == '__main__':
    args = setup(run_args.config_file)
    args.model = run_args.model
    args.gpu = run_args.gpu

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.__dict__['pareto_models']['exits_checkpoint_path'] = '/home/hbouzidi/hbouzidi/AttentiveNAS/saved_models/attentive_nas_model_'+args.model+'_'+args.dataset+'_exits/attentive_nas.pth.tar'
    args.__dict__['active_subnet'] = args.__dict__['pareto_models'][args.model]

    train_loader, val_loader, train_sampler = build_data_loader(args)

    ## init static attentivenas model with weights inherited from the supernet 
    n_exits = args.__dict__['active_subnet']['num_ee'] 
    model = models.model_factory.create_model(args)
    model.to(args.gpu)
    model.eval()
    
    # bn running stats calibration following Slimmable (https://arxiv.org/abs/1903.05134)
    # please consider trying a different random seed if you see a small accuracy drop
    with torch.no_grad():
        model.reset_running_stats_for_calibration()
        for batch_idx, (images, _) in enumerate(train_loader):
            if batch_idx >= args.post_bn_calibration_batch_num:
                break
            images = images.cuda(args.gpu, non_blocking=True)
            model(images)  #forward only

    torch.cuda.empty_cache()

    model.eval()
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss().cuda()
        from evaluate.exit_eval import validate_eex_subnet
        acc1_list, acc5_list, loss_list, conf_list, path_list, entropy_list, correct_list = validate_eex_subnet(val_loader, model, criterion, n_exits, args)  # OD: flops,params were other return variables

    for i in range(n_exits+1):
        entropy_list[i] = [round(x.item(),4) for x in entropy_list[i]]
        correct_list[i] = [x.item() for x in correct_list[i]]
        if path_list[i] and type(path_list[i]) is not str:
            path_list = [x.item() for x in path_list[i]]

    # Exit performances saving:
    header = ['Exit', 'TOP-1 Acc', 'TOP-5 Acc', 'Loss']

    with open('./'+args.model+'_'+args.dataset+'exit_perf.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(n_exits+1):
            writer.writerow(['Exit_'+str(i+1), acc1_list[i], acc5_list[i], loss_list[i]])

    # Entropy results saving:
    header = ['Exit', 'Sample_path', 'Entropy', 'Correct', 'Confidence']

    rows_exits = []
    for i in range(n_exits+1):
        rows_exits.append([path_list[i], entropy_list[i], correct_list[i], conf_list[i]])

    with open('./'+args.model+'_'+args.dataset+'entropy_table.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, row in enumerate(rows_exits, start=0):
            for j in range(len(row[1])):
                writer.writerow(['Exit_'+str(i+1),'Sample_'+str(j+1) , row[1][j], row[2][j], row[3][j].item()])
