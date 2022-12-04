import numpy as np
import random
import copy

class EExDVFSSearchSpace:
    def __init__(self, n_blocks=12):

        # **************************** EEx related decision variables *****************************#
        self.n_blocks = n_blocks # number of potential layers for earl-exit
        self.n_exits = [i for i in range(1, self.n_blocks-5)]
        self.pos_exits  = [i for i in range(5, self.n_blocks)]

        # **************************** DVFS parameters on the Edge GPU (depending on the HW-platform) *****************************#
        self.cpu = [115200, 192000, 268800, 345600, 422400, 499200, 576000, 652800, 729600, 806400, 883200, 960000, 1036800, 1113600, 1190400, 
                    1267200, 1344000, 1420800, 1497600, 1574400, 1651200, 1728000, 1804800, 1881600, 1958400, 2035200, 2112000, 2188800, 2265600]
        self.gpu = [114750000, 216750000, 318750000, 420750000, 522750000, 624750000, 675750000, 828750000, 905250000, 1032750000, 1198500000, 
                    1236750000, 1338750000, 1377000000]
        self.emc = [204000, 408000000, 665600000, 800000000, 1065600000, 1331200000, 1600000000, 1866000000, 2133000000]

        self.cfg_candidates = {
            'n_blocks': self.n_blocks ,
            'n_exits': self.n_exits,
            'pos_exits': self.pos_exits,
            'cpu': self.cpu,
            'gpu': self.gpu,
            'emc': self.emc
        }

    #********************************************** Architectures sampling ************************************************#

    def sample_all(self, n_samples=1):
        
        data = []
        for _ in range(n_samples):
            
            n_exits = random.choice(self.cfg_candidates['n_exits'])
            pos_exits = np.zeros(n_exits, dtype = object)

            for i in range(n_exits):
                while(True):
                    pos = random.choice(self.cfg_candidates['pos_exits'])
                    if(not pos in pos_exits):
                        pos_exits[i] = pos
                        break

            cpu_ = random.choice(self.cfg_candidates['cpu'])
            gpu_ = random.choice(self.cfg_candidates['gpu'])
            emc_ = random.choice(self.cfg_candidates['emc'])

            data.append({'n_exits': n_exits, 'pos_exits': pos_exits, 'cpu': cpu_, 'gpu': gpu_, 'emc': emc_})

        return data


    def initialize_all(self, n_doe):        
        return self.sample_all(n_samples=n_doe)


    #*************************************************** Mutation and Crossover ******************************************************#

    def mutate_and_reset(self, cfg, prob=0.1):
        cfg = copy.deepcopy(cfg)
        pick_another = lambda x, candidates: x if len(candidates) == 1 else random.choice([v for v in candidates if v != x])
        
        # Sample a DVFS configuration
        r = random.random()
        if r < prob:
            cfg['cpu'] = pick_another(cfg['cpu'], self.cfg_candidates['cpu'])
        
        if r < prob:
            cfg['gpu'] = pick_another(cfg['gpu'], self.cfg_candidates['gpu'])

        if r < prob:
            cfg['emc'] = pick_another(cfg['emc'], self.cfg_candidates['emc'])

        # Sample exits number and position, we only alter the exits positions
        r = random.random()
        for i in range(cfg['n_exits']):
            r = random.random()
            if r < prob:
                cfg['pos_exits'][i] = pick_another(cfg['pos_exits'][i], self.cfg_candidates['pos_exits'])

        # Eliminate duplicated from the obtained exit positions
        cfg['pos_exits'] = list(dict.fromkeys(cfg['pos_exits']))
        cfg['n_exits'] = len(cfg['pos_exits'])

        return cfg
    
    def crossover_and_reset(self, cfg1, cfg2, p=0.5):
        def _cross_helper(g1, g2, prob):
            assert type(g1) == type(g2)
            if isinstance(g1, int):
                return g1 if random.random() < prob else g2
            elif isinstance(g1, list):
                return [v1 if random.random() < prob else v2 for v1, v2 in zip(g1, g2)]
            else:
                raise NotImplementedError

        cfg = {}
        
        # sample a DVFS configuration
        cfg['cpu'] = cfg1['cpu'] if random.random() < p else cfg2['cpu']
        cfg['gpu'] = cfg1['gpu'] if random.random() < p else cfg2['gpu']
        cfg['emc'] = cfg1['emc'] if random.random() < p else cfg2['emc']

        # Sample exits number and position
        cfg['n_exits'] = cfg1['n_exits'] if random.random() < p else cfg2['n_exits']
        cfg['pos_exits'] = cfg1['pos_exits'] if cfg['n_exits']==cfg1['n_exits'] else cfg2['pos_exits']
        
        for i in range(cfg['n_exits']):
            if(i < len(cfg1['pos_exits']) and i < len(cfg2['pos_exits'])):
                cfg['pos_exits'][i] = _cross_helper(cfg1['pos_exits'][i], cfg2['pos_exits'][i], p)

        # Eliminate duplicated from the obtained exit positions
        cfg['pos_exits'] = list(dict.fromkeys(cfg['pos_exits']))
        cfg['n_exits'] = len(cfg['pos_exits'])

        return cfg


if __name__ == '__main__':

    eex_dvfs_search_space = EExDVFSSearchSpace()
    rows = []
    init_ = eex_dvfs_search_space.initialize_all(10)
    print(init_[3])
    
