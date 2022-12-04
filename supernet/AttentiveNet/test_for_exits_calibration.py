# Test script to obtain the entropy levels of the different data samples on multiple exits

# Test script to conduct the calibration process for models by estimating the temperature scale values

import argparse
import math
import os
import random
import time
import argparse
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
import models
from data.data_loader import build_data_loader, build_val_data_loader
from utils.progress import AverageMeter, ProgressMeter, accuracy
from utils.calibration import temp_estimator

parser = argparse.ArgumentParser(description='Test AttentiveNas Models')
parser.add_argument('--config-file', default='./configs/eval_attentive_nas_models.yml')
parser.add_argument('--model', default='a0', type=str, choices=['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6'])
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--temp_step', default=5, type=int, help='stepsize for temperature sweeps')
parser.add_argument('--n_bins', default=5, type=int, help='Number of confidence bins')
parser.add_argument('--temp_min', default=0, type=float, help='start temperature for sweep')
parser.add_argument('--temp_max', default=5000, type=int, help='end temperature for sweep')
parser.add_argument('--loader', default='train', type=str, choices=['train', 'val'])
parser.add_argument('-single', action='store_true', help='single value evalation of \'temp_min\'')
parser.add_argument('-save', action='store_true', help='if single evaluations are to be stored')

run_args = parser.parse_args()

if __name__ == '__main__':
    args = setup(run_args.config_file)
    args.model = run_args.model
    args.gpu = run_args.gpu
    args.loader = run_args.loader
    args.temp_step = run_args.temp_step
    args.n_bins = run_args.n_bins
    args.temp_min = run_args.temp_min
    args.temp_max = run_args.temp_max
    args.single = run_args.single
    args.save = run_args.save
    assert args.temp_step < args.temp_max

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
        for batch_idx, (images, _, _) in enumerate(train_loader):
            if batch_idx >= args.post_bn_calibration_batch_num:
                break
            images = images.cuda(args.gpu, non_blocking=True)
            model(images)  #forward only

    torch.cuda.empty_cache()

    model.eval()
    dataloader = train_loader if args.loader == 'train' else val_loader
    temp_object = temp_estimator(args, model, dataloader, temp_step=args.temp_step, n_bins=args.n_bins, temp_min=args.temp_min, temp_max=args.temp_max, n_exits=n_exits)

    header = ['Temp', 'ECE']

    if args.single:
    	temp, ECE = temp_object.evaluate()
        
        print(f"ECE: {ECE} @temp: {temp}")

        if args.save:
		
            with open('dataset_analysis/temp_calibration/'+args.dataset+'/'+args.model+'_calibration_table_'+args.loader+'_single.csv', 'a') as f:
		    	# Append to existing
		
                writer = csv.writer(f)
		
                writer.writerow([temp, ECE])
    	
        exit()

    best_temp, best_ECE = temp_object.sweep()
    
    temp_list, ECE_list = temp_object.temp_list, temp_object.ECE_list

    rows = zip(temp_list, ECE_list)

    with open('dataset_analysis/temp_calibration/'+args.dataset+'/'+args.model+'_calibration_table_'+args.loader+'_sweep.csv', 'w') as f:
    	# Overwrites existing
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)