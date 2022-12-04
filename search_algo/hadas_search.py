import torch
import argparse
import random
import math
import threading
import numpy as np
from supernet.AttentiveNet import models
from supernet.AttentiveNet.utils.config import setup
from search_space.eex_dvfs_search_space import NASSearchSpace, EExDVFSSearchSpace
from utils_opt import RankAndCrowdingSurvivalOuter, RankAndCrowdingSurvivalInner
from utils_eval import remote_eval_nas_err, remote_eval_nas_hw, read_results_nas_err, read_results_nas_hw, \
                       remote_eval_dvfs, remote_eval_exits, read_eex_scores, read_results_dvfs

parser = argparse.ArgumentParser(description='Run the optimization framework of HADAS for optimal DyNNs')
parser.add_argument('--config-file', default='./config.yml')
run_args = parser.parse_args()

def inner_optimization_engine(backbone, args):
    n_blocks = 0
    for d in backbone['depth']: n_blocks +=d
    ss_eex_dvfs = EExDVFSSearchSpace(n_blocks=n_blocks)
    parent_popu_inner = ss_eex_dvfs.initialize_all(args.evo_search_inner.parent_popu_size)
  
    ## Run the evolutionary search for the inner optimization engine (IOE)
    pareto_inner = {}
    for evo in range(args.evo_search_inner.evo_iter):           

        thread1 = threading.Thread(target = remote_eval_dvfs, args=(backbone['id'], parent_popu_inner, ))
        thread2 = threading.Thread(target = remote_eval_exits, args=(backbone['id'], parent_popu_inner, ))
        
        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

        err = read_eex_scores(backbone['id']) # for each exit
        lat, energy = read_results_dvfs(backbone['id']) # latency and energy for each exit

        #save_results_inner(backbone['id'], parent_popu_inner, err, lat, energy)

        for i in range(len(parent_popu_inner)):
            parent_popu_inner[i]['err'] = np.mean(err[i])
            parent_popu_inner[i]['latency'] = np.mean(lat[i])
            parent_popu_inner[i]['energy'] = np.mean(energy[i])

        ## Survival selection for the inner optimzation based on eex_dvfs performances:
        nb_survival = math.ceil(len(parent_popu_inner)*args.evo_search_inner.survival_ratio)
        survivals = RankAndCrowdingSurvivalInner(pop=parent_popu_inner, n_survive=nb_survival)

        # Update the Pareto frontier
        for i, cfg in enumerate(survivals, start=0):
            pareto_inner[i] = cfg
            
        ## Create the next batch of eex_dvfs to evaluate
        parent_popu_inner = []

        ## Crossover 
        for _ in range(args.evo_search_inner.crossover_size):
            cfg1 = random.choice(list(pareto_inner.values()))
            cfg2 = random.choice(list(pareto_inner.values()))
            cfg = ss_eex_dvfs.crossover_and_reset(cfg1, cfg2, args.evo_search_inner.crossover_prob)
            parent_popu_inner.append(cfg)

        ## Mutation
        for _ in range(args.evo_search_inner.mutate_size):          
            old_cfg = random.choice(list(pareto_inner.values()))
            cfg = ss_eex_dvfs.mutate_and_reset(old_cfg, prob=args.evo_search_inner.mutate_prob)
            parent_popu_inner.append(cfg)

def outer_optimization_engine(gpu, args):

    args.gpu = gpu  # local rank, local machine cuda id 
    torch.cuda.set_device(args.gpu)

    ## Create the supernet for backbones sampling
    supernet = models.model_factory.create_model(args)
    model.cuda(args.gpu)
           
    ## Population initialization
    ss_nas = NASSearchSpace()
    parent_popu_outer = ss_nas.initialize_all(args.evo_search_inner.parent_popu_size)

    ## Run the evolutionary search for the outer optimization engine (OOE)
    pareto_outer = {}
    for evo in range(args.evo_search.evo_iter):   
        
        thread1 = threading.Thread(target = remote_eval_nas_err, args=(parent_popu_outer, ))
        thread2 = threading.Thread(target = remote_eval_nas_hw, args=(parent_popu_outer, ))
        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()     

        err = read_results_nas_err(path='tmp')
        lat, energy = read_results_nas_hw(path='tmp')

        for i in range(len(parent_popu_outer)):
            parent_popu_outer[i]['err'] = err[i]
            parent_popu_outer[i]['latency'] = lat[i]
            parent_popu_outer[i]['energy'] = energy[i]

        #save_results_outer(parent_popu_outer, err, lat, energy)

        ## First Survival selection --> for the inner optimzation based on nas performances:
        nb_survival = math.ceil(len(parent_popu_outer)*args.evo_search_outer.survival_ratio)
        outer_survivals = RankAndCrowdingSurvivalOuter(pop=parent_popu_outer, n_survive=nb_survival)
        
        for backbone in outer_survivals:
            inner_optimization_engine(backbone, parent_popu_outer[int(backbone['id'])], args) 

        ## Second Survival selection --> for the outer optimzation based on nas_eex_dvfs performances:
        nb_survival = math.ceil(len(parent_popu_outer)*args.evo_search_outer.survival_ratio/2)
        inner_survivals = RankAndCrowdingSurvivalOuter(pop=outer_survivals, n_survive=nb_survival)   
            
        # Update the Pareto frontier
        for i, cfg in enumerate(inner_survivals, start=0):
            pareto_outer[i] = cfg

        ## Create the next batch of backbone netwokrs to evaluate
        parent_popu_outer = []

        ## Crossover
        for _ in range(args.evo_search.crossover_size):
            cfg1 = random.choice(list(pareto_outer.values()))
            cfg2 = random.choice(list(pareto_outer.values()))
            cfg = supernet.module.crossover_and_reset(cfg1, cfg2)
            parent_popu_outer.append(cfg)
        
        ## Mutation
        for _ in range(args.evo_search.mutate_size):          
            old_cfg = random.choice(list(pareto_outer.values()))
            cfg = supernet.module.mutate_and_reset(old_cfg, prob=args.evo_search.mutate_prob)
            parent_popu_outer.append(cfg)   
    


if __name__ == '__main__':
    
    args = setup(run_args.config_file)
    outer_optimization_engine(gpu=run_args.gpu, args=args)
    
