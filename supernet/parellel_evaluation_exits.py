import argparse
import random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from utils.config import setup
from utils.factory import read_eex_population, save_eex_evaluation
import utils.comm as comm

from AttentiveNAS import models
from AttentiveNAS.data.data_loader import build_data_loader
from AttentiveNAS import train_exit_blocks as eval_exit

parser = argparse.ArgumentParser(description='Train and Evaluate Dynamic Neural Networks with early-exiting')
parser.add_argument('--config-file', default='./config.yml')
parser.add_argument('--machine-rank', default=0, type=int, help='machine rank, distributed setting')
parser.add_argument('--num-machines', default=16, type=int, help='number of nodes, distributed setting')
parser.add_argument('--dist-url', default="", type=str, help='init method, distributed setting')
parser.add_argument('--seed', default=1, type=int, help='default random seed')
run_args = parser.parse_args()

def eval_worker(gpu, ngpus_per_node, args):
    
    args.gpu = gpu  # local rank, local machine cuda id
    args.local_rank = args.gpu
    args.batch_size = args.batch_size_per_gpu

    global_rank = args.gpu + args.machine_rank * ngpus_per_node
      
    dist.init_process_group(
        backend=args.dist_backend, 
        init_method=args.dist_url,
        world_size=args.world_size, 
        rank=global_rank
    )
    
    # Synchronize is needed here to prevent a possible timeout after calling
    comm.synchronize()

    args.rank = comm.get_rank() # global rank
    torch.cuda.set_device(args.gpu)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Build the supernet
    model = models.model_factory.create_model(args)
    model.cuda(args.gpu)
    model = comm.get_parallel_model(model, args.gpu) #local rank
    
    ## Load training/validation datasets, train_sampler: if distributed
    train_loader, val_loader, _ =  build_data_loader(args)

    assert args.resume

    # Reloading the pretrained supernet for weights inheritance
    model.module.load_weights_from_pretrained_models(args.resume)

    parent_popu = read_eex_population(path='tmp/')
    
    # Reading the population that needs to be evaluated
    for idx, cfg in enumerate(parent_popu, start=0):
        cfg['net_id'] = f'net_{idx % args.world_size}_{idx}'

    # DyNNs to be evaluated on GPU {args.rank}
    # Send each DyNN to a GPU for parallel early-exit branches training
    my_subnets_to_be_evaluated = {}
    n_evaluated = len(parent_popu) // args.world_size * args.world_size

    for cfg in parent_popu[:n_evaluated]:
        if cfg['net_id'].startswith(f'net_{args.rank}_'):
            my_subnets_to_be_evaluated[cfg['net_id']] = cfg
    
    # Aggregating all evaluation results from all GPUs
    eval_results = eval_exit.train(
        my_subnets_to_be_evaluated,
        train_loader,
        val_loader,
        model,
        args, 
    )  

    save_eex_evaluation(parent_popu, eval_results, path='tmp/exits_lut')

    
if __name__ == '__main__':
    args = setup(run_args.config_file)
    args.dist_url = run_args.dist_url
    args.machine_rank = run_args.machine_rank
    args.num_nodes = run_args.num_machines

    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.num_nodes
        assert args.world_size > 1, "only support DDP settings"
        mp.spawn(eval_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        raise NotImplementedError