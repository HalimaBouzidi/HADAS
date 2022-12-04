import csv
import subprocess
import paramiko

## The following functions are to be re-implemented according to the user’s experimental setup
## Key API to establish remote evaluation is: Paramiko with ssh tunnel to remotely run the evaluation scripts

def read_nas_population(path='tmp/'):
    pass

def read_eex_population(path='tmp/exits_lut'):
    pass

def save_nas_evaluation(parent_popu, eval_results, path='tmp/'):
    pass

def save_nas_evaluation(parent_popu, eval_results, path='tmp/exits_lut'):
    pass