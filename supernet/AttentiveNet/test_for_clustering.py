# Test script to obtain the entropy levels of the different data samples

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
from data.data_loader import build_data_loader, build_val_data_loader
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

    args.__dict__['active_subnet'] = args.__dict__['pareto_models'][args.model]

    train_loader, val_loader, train_sampler = build_data_loader(args)

    ## init static attentivenas model with weights inherited from the supernet 
    model = models.model_factory.create_model(args)
    model.to(args.gpu)
    model.eval()
    
    # bn running stats calibration following Slimmable (https://arxiv.org/abs/1903.05134)
    # please consider trying a different random seed if you see a small accuracy drop
    with torch.no_grad():
        model.reset_running_stats_for_calibration()
        for batch_idx, (images, _, _) in enumerate(train_loader):
            if batch_idx >= args.post_bn_calibration_batch_num:
                break
            images = images.cuda(args.gpu, non_blocking=True)
            model(images)  #forward only

    torch.cuda.empty_cache()

    model.eval()
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss().cuda()
        from evaluate.imagenet_eval import validate_one_subnet
        acc1, acc5, loss, path_list, entropy_list, correct_list = validate_one_subnet(val_loader, model, criterion, args)  # OD: flops,params were other return variables
        print(acc1, acc5)#, flops, params)

    entropy_list = [round(x.item(),4) for x in entropy_list]
    correct_list = [x.item() for x in correct_list]
    if type(path_list) is not str:
        path_list = [x.item() for x in path_list]

    header = ['Sample_path', 'Entropy', 'Correct']
    rows = zip(path_list, entropy_list, correct_list)

    with open('dataset_analysis/entropy/'+args.dataset+'/'+args.model+'_entropy_table_'+args.dataloader+'.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)