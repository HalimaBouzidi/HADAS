from mimetypes import init
from tracemalloc import start
import numpy as np
from numpy.core.defchararray import decode
from torchinfo import summary
import re

class SearchSpace:
    def __init__(self):

        self.num_blocks = 7  # number of blocks (fixed by the macro-architecture)

        # ****************************** Block width: Output channels size ****************************** #
        self.width = [[16, 24], [16, 24], [24, 32], [32, 40], [64, 72], [112, 120, 128], [192, 200, 208, 216], [216, 224], [1792, 1984]]

        # ****************************** Block depth: number of layers ****************************** #
        self.depth = [[1,2], [3, 4, 5], [3, 4, 5, 6], [3, 4, 5, 6], [3, 4, 5, 6, 7, 8], [3, 4, 5, 6, 7, 8], [1,2]]

        # ************************ Common parameters: kernel size and expansion ratio ****************** #
        self.resolution = [192, 224, 256, 288]
        self.kernel_size = [3, 5] 
        self.exp_ratio = [4, 5, 6]

    def bounds_arch(self, n_var=28):
        lb = np.zeros(n_var, dtype = object)
        ub = np.zeros(n_var, dtype = object)

        ub[0] = len(self.resolution) -1

        for i in range(0, len(self.width)):
            ub[i+1] = len(self.width[i]) -1

        for i in range(0, len(self.depth)):
            ub[i+10] = len(self.depth[i]) -1

        for i in range(0, self.num_blocks):
            ub[i+17] = len(self.kernel_size) -1

        for i in range(0, 4):
            ub[i+24] = len(self.exp_ratio) -1

        return lb, ub

    #********************************************** Architectures sampling ************************************************#

    def sample_all(self, n_samples=1, nb=None, res=None, width=None, depth=None, ks=None, er=None, cpu=None, gpu=None, emc=None):
        
        nb = self.num_blocks if nb is None else nb
        res = self.resolution if res is None else res
        width = self.width if width is None else width
        depth = self.depth if depth is None else depth        
        ks = self.kernel_size if ks is None else ks
        er = self.exp_ratio if er is None else er

        data = []
        for n in range(n_samples):
            
            resolution = np.random.choice(res)

            width_ = []
            depth_ = []

            for i in range(0, len(width)):
                width_.append(np.random.choice(width[i]))

            for i in range(0, len(depth)):
                depth_.append(np.random.choice(depth[i]))   

            kernel_size = np.random.choice(ks, size=nb, replace=True).tolist()
            expand_ratio = np.random.choice(er, size=4, replace=True).tolist()

            data.append({'resolution': resolution, 'width': width_, 'depth': depth_,  'kernel_size': kernel_size, 'expand_ratio': expand_ratio})

        return data


    def initialize_all(self, n_doe):        
        return self.sample_all(n_samples=n_doe)

    #**************************************************** Architectures Encoding *****************************************************#

    def encode_all(self, config):
        x = []
        x.append(np.argwhere(config['resolution'] == np.array(self.resolution))[0, 0])

        x += [np.argwhere(_x == np.array(self.width[idx]))[0, 0] for idx, _x in enumerate(config['width'], start=0)]
        x += [np.argwhere(_x == np.array(self.depth[idx]))[0, 0] for idx, _x in enumerate(config['depth'], start=0)]
        
        x += [np.argwhere(_x == np.array(self.kernel_size))[0, 0] for _x in config['kernel_size']]
        x += [np.argwhere(_x == np.array(self.exp_ratio))[0, 0] for _x in config['expand_ratio']]

        return x

    #**************************************************** Architectures Decoding *****************************************************#
    
    def decode_all(self, x):   
        width, depth, kernel_size, expand_ratio = [], [], [], []

        resolution = self.resolution[int(x[0])]

        for i in range(0, 9): 
            width.append(self.width[i][int(x[i+1])])

        for i in range(0, 7): 
            depth.append(self.depth[i][int(x[i+10])])

        for i in range(0, 7):
            kernel_size.append(self.kernel_size[int(x[i+17])])
            
        expand_ratio.append(1)
        for i in range(0, 4):
            expand_ratio.append(self.exp_ratio[int(x[i+24])])
        expand_ratio.append(6)
        expand_ratio.append(6)
        
        arch = {'resolution': resolution, 'width': width, 'depth': depth,  'kernel_size': kernel_size, 'expand_ratio': expand_ratio}

        return arch

    def encode_to_x(self, config):

        expand = list(np.array(config[4].split('_')).astype(int))
        expand_ = [expand[1], expand[2], expand[3], expand[4]]

        return {'resolution': int(config[0]), 
                'width': list(np.array(config[1].split('_')).astype(int)), 
                'depth': list(np.array(config[2].split('_')).astype(int)), 
                'kernel_size': list(np.array(config[3].split('_')).astype(int)), 
                'expand_ratio': expand_,
                }

if __name__ == '__main__':

    from joblib import load
    search_space = SearchSpace()
    rows = []
    init_ = search_space.initialize_all(20)
    print(init_[3])
    print(search_space.encode_all(init_[3]))
    decod_ = search_space.decode_all(search_space.encode_all(init_[3]))
    print(decod_)
    # Loading the accuracy predictor
    predictor = load('./AttentiveNet/acc_predictor.joblib')
    res = [decod_['resolution']]
    for k in ['width', 'depth', 'kernel_size', 'expand_ratio']:
        res += decod_[k]
    input = np.asarray(res).reshape((1, -1))
    acc = predictor.predict(input)
    print(acc)
