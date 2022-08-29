import os, sys
import yaml
import numpy as np
from pymoo.factory import get_performance_indicator

def find_pareto_front(Y, return_index=False):
    '''
    Find pareto front (undominated part) of the input performance data.
    '''
    if len(Y) == 0: return np.array([])
    sorted_indices = np.argsort(Y.T[0])
    pareto_indices = []
    for idx in sorted_indices:
        # check domination relationship
        if not (np.logical_and((Y <= Y[idx]).all(axis=1), (Y < Y[idx]).any(axis=1))).any():
            pareto_indices.append(idx)
    pareto_front = Y[pareto_indices].copy()

    if return_index:
        return pareto_front, pareto_indices
    else:
        return pareto_front

def calc_hypervolume(pfront, ref_point):
    '''
    Calculate hypervolume of pfront based on ref_point
    '''
    hv = get_performance_indicator('hv', ref_point=ref_point)
    return hv.calc(pfront)

def get_result_dir(args):
    '''
    Get directory of result location (result/problem/subfolder/algo/seed/)
    '''
    top_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)), 'result')
    exp_name = '' if args.exp_name is None else '-' + args.exp_name
    algo_name = args.algo + exp_name
    result_dir = os.path.join(top_dir, args.problem, args.subfolder, algo_name, str(args.seed))
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


def save_args(general_args, framework_args):
    '''
    Save arguments to yaml file
    '''
    all_args = {'general': vars(general_args)}
    all_args.update(framework_args)

    result_dir = get_result_dir(general_args)
    args_path = os.path.join(result_dir, 'args.yml')

    os.makedirs(os.path.dirname(args_path), exist_ok=True)
    with open(args_path, 'w') as f:
        yaml.dump(all_args, f, default_flow_style=False, sort_keys=False)


def setup_logger(args):
    '''
    Log to file if needed
    '''
    logger = None

    if args.log_to_file:
        result_dir = get_result_dir(args)
        log_path = os.path.join(result_dir, 'log.txt')
        logger = open(log_path, 'w')
        sys.stdout = logger
    
    return logger
