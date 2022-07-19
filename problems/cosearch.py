from abc import ABC, abstractmethod
import os
import subprocess
import numpy as np
import autograd.numpy as anp
import csv
import threading
import paramiko
from joblib import load
from .problem import Problem
from .search_space import SearchSpace
from .AttentiveNet.model_eval import subnet_acc_eval, subnet_flops_eval
from .AttentiveNet.utils.config import setup
from .AttentiveNet.data.data_loader import build_data_loader

directory = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
remote_dir_edge = "hellojetsonagx@agx:~/workspace/hw_dynamic_nas/" # to change according to the jetson

#********************************************* NAS problem definition ************************************************#

class nas_problem(Problem):

    def __init__(self, platform=''):
        
        # We define 28 decision variables to encode the architecture
        # We define 02 objectives (prediction error and energy consumption)
        super().__init__(n_var=28, n_obj=2, n_constr=0)

        self.ss = SearchSpace()
        self.xl = np.array(self.ss.bounds_arch()[0]) 
        self.xu = np.array(self.ss.bounds_arch()[1]) 
        self.platform = platform

    def _evaluate(self, x, out, *args, requires_F=True, **kwargs):
        if requires_F:
            out['F'] = np.column_stack([*self._evaluate_F(x)])

    @abstractmethod
    def _evaluate_F(self, x):
        pass

    def _calc_pareto_front(self):
        pass

class HW_NAS(nas_problem):

    def _evaluate_F(self, x):
        rows = []       
        with open(directory+'/tmp/current_pop.csv', 'w') as f:
            writer = csv.writer(f, delimiter=' ')
            for config in x:
                arch = self.ss.decode_all(config)
                row = [arch['resolution'], '_'.join(str(e) for e in arch['width']), '_'.join(str(e) for e in arch['depth']),
                '_'.join(str(e) for e in arch['kernel_size']), '_'.join(str(e) for e in arch['expand_ratio'])] 
                rows.append(row)
                writer.writerow(row)
             
        thread1 = threading.Thread(target = objective_sw_pred, args=(x, len(x),)) # fast evaluation of accuracy
        #thread1 = threading.Thread(target = objective_sw_val, args=(x, len(x),)) # full evaluation of accuracy

        thread2 = threading.Thread(target = objective_hw_proxy, args=(x, len(x),)) # proxy evaluation of hw efficiency
        #thread2 = threading.Thread(target = objective_hw_direct, args=(x, len(x),)) # direct evaluation of hw efficiency

        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()
  
        # Read the evelauation results
        f_all1 = read_results_error()
        f_all2 = read_results_flops()

        # Save the obtained results in the archive
        save_results(rows, f_all1, f_all2)
        
        return f_all1, f_all2

    def initialize_pop(self, n_samples):
        return self.ss.initialize_all(n_doe=n_samples)

    def encode_sample(self, config):
        return self.ss.encode_all(config)

    def decode_sample(self, x):
        return self.ss.decode_all(x)

#****************************************** Objectives evaluation **********************************************#

# Fast accuracy evaluation with the pretrained prediction model provided by AttentiveNAS
def objective_sw_pred(x, n_samples=None):
    ss = SearchSpace()
    predictor = load(directory+'/problems/AttentiveNet/acc_predictor.joblib')
    with open(directory+'/tmp/current_pop_err.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for config in x:
            arch = ss.decode_all(config)
            res = [arch['resolution']]
            for k in ['width', 'depth', 'kernel_size', 'expand_ratio']:
                res += arch[k]
            input = np.asarray(res).reshape((1, -1))
            err = 100 - predictor.predict(input)[0] ## TOP-1 prediction error
            row = [str(arch['resolution']), '_'.join(str(e) for e in arch['width']), '_'.join(str(e) for e in arch['depth']),
                '_'.join(str(e) for e in arch['kernel_size']), '_'.join(str(e) for e in arch['expand_ratio'])] + [str(err)]
            writer.writerow(row)


# Full accuracy evaluation on the valdation dataset
def objective_sw_val(x, n_samples=None):
    setup_file = directory+'/problems/AttentiveNet/configs/eval_attentive_nas_models.yml'
    args = setup(setup_file)
    train_loader, val_loader, train_sampler = build_data_loader(args)

    ss = SearchSpace()
    with open(directory+'/tmp/current_pop_err.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for config in x:
            arch = ss.decode_all(config)
            err = 100 - subnet_acc_eval(arch, train_loader, val_loader, args) ## TOP-1 prediction error
            row = [str(arch['resolution']), '_'.join(str(e) for e in arch['width']), '_'.join(str(e) for e in arch['depth']),
                '_'.join(str(e) for e in arch['kernel_size']), '_'.join(str(e) for e in arch['expand_ratio'])] + [str(err)]
            writer.writerow(row)

# Fast evaluation of the hardware efficiency with a proxy metric (FLOPs)
def objective_hw_proxy(x, n_samples=None): 

    setup_file = directory+'/problems/AttentiveNet/configs/eval_attentive_nas_models.yml'
    args = setup(setup_file)
    ss = SearchSpace()

    with open(directory+'/tmp/current_pop_flops.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for config in x:
            arch = ss.decode_all(config)
            flops = subnet_flops_eval(arch, args)
            row = [str(arch['resolution']), '_'.join(str(e) for e in arch['width']), '_'.join(str(e) for e in arch['depth']),
                '_'.join(str(e) for e in arch['kernel_size']), '_'.join(str(e) for e in arch['expand_ratio'])] + [str(flops)]
            writer.writerow(row)

# Direct evaluation of the hardware efficiency on the hardware device (Edge GPU)
def objective_hw_direct(x, n_samples=None): 
    hostname = "192.168.55.1"
    username = "hellojetsonagx"
    privatekeyfile = "/home/halima/.ssh/id_rsa"
    main_dir = "/home/hellojetsonagx/workspace/imgclsmob-master/pytorch/cosearch/"

    # Send the sampled population for evaluation to the Edge GPU (AGX Xavier)
    process = subprocess.run(['scp', directory+'/tmp/current_pop.csv', remote_dir_edge+'tmp/'], 
                            stdout=subprocess.PIPE, universal_newlines=True)
    err = process.stderr
    if err:
        print(err)

    # Launch the population evaluation job on the Edge GPU (AGX Xavier)
    mkey = paramiko.RSAKey.from_private_key_file(privatekeyfile)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(hostname=hostname, username=username, pkey=mkey)
    except:
        print("[!] Cannot connect to the SSH Server")
        exit()

    command_1 = "/bin/bash "+main_dir+"script_hw.sh"
    stdin, stdout, stderr = client.exec_command(command_1)
    err = stderr.read().decode()
    if err:
        print(err)
    client.close() 
        
    # Now, send back the evaluation results to local host
    process = subprocess.run(['scp', remote_dir_edge+'tmp/current_pop_energy.csv', directory+'/tmp/'], 
                            stdout=subprocess.PIPE, universal_newlines=True)

    err = process.stderr
    if err:
        print(err)
    
#****************************************** Results saving and reading **********************************************#

def read_results_error():
    error = []
    with open(directory+'/tmp/current_pop_err.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            err = row[5]
            if(err != 'None'):
                error.append(float(err))    
    return np.asarray(error)

def read_results_flops(path='tmp'):
    flops = []
    with open(directory+'/tmp/current_pop_flops.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            fp = row[5]
            if(fp != 'None'):
                flops.append(float(fp))
    return np.asarray(flops)

def read_results_energy(path='tmp'):
    energy = []
    with open(directory+'/tmp/current_pop_energy.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            eg = row[5]
            if(eg != 'None'):
                energy.append(float(eg))

    return np.asarray(energy)

def save_results(x, obj1, obj2):
    with open(directory+'/tmp/archive_pop.csv', 'a') as f:
        writer = csv.writer(f, delimiter=',')
        for idx, config in enumerate(x, start=0):
            row = config + [obj1[idx]] + [obj2[idx]]
            writer.writerow(row)  


#****************************************************************************************************************************#

if __name__ == '__main__':

    search_space = SearchSpace()
    init_ = search_space.initialize_all(20)
    decod_ = search_space.decode_all(search_space.encode_all(init_[3]))
    print(decod_)

    # Test the fast evaluation with the accuracy predictor
    predictor = load('./AttentiveNet/acc_predictor.joblib')
    res = [decod_['resolution']]
    for k in ['width', 'depth', 'kernel_size', 'expand_ratio']:
        res += decod_[k]
    input = np.asarray(res).reshape((1, -1))
    acc = predictor.predict(input)
    print(acc)

    # Test the full evaluation on the validation dataset
    setup_file = directory+'/problems/AttentiveNet/configs/eval_attentive_nas_models.yml'
    args = setup(setup_file)
    train_loader, val_loader, train_sampler = build_data_loader(args)
    print(subnet_acc_eval(decod_, train_loader, val_loader, args))
    print(subnet_flops_eval(decod_, args))

