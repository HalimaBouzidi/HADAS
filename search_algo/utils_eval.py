import csv
import subprocess
import paramiko

## The following functions are to be re-implemented according to the user’s experimental setup
## Key API to establish remote evaluation is: Paramiko with ssh tunnel to remotely run the evaluation scripts

def remote_eval_nas_err(population):
    pass

def remote_eval_nas_hw(population):
    pass

def remote_eval_dvfs(backbone_id, population):
    pass

def remote_eval_exits(backbone_id, population):
    pass

def read_eex_scores(backbone_id, path='tmp'):
    pass

def read_results_dvfs(backbone_id, path='tmp'):
    pass

def read_results_err(path='tmp'):
    pass

def read_results_hw(path='tmp'):
    pass


def save_results_outer(population, err, latency, energy):
    pass 

def save_results_inner(backbone_id, population, err, latency, energy):
    pass