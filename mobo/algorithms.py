from .mobo import MOBO

'''
High-level algorithm specifications by providing config
'''

class DGEMO(MOBO):
    '''
    DGEMO
    '''
    config = {
        'surrogate': 'gp',
        'acquisition': 'identity',
        'solver': 'discovery',
        'selection': 'dgemo',
    }


class TSEMO(MOBO):
    '''
    TSEMO
    '''
    config = {
        'surrogate': 'ts',
        'acquisition': 'identity',
        'solver': 'nsga2',
        'selection': 'hvi',
    }


class USEMO_EI(MOBO):
    '''
    USeMO, using EI as acquisition
    '''
    config = {
        'surrogate': 'gp',
        'acquisition': 'ei',
        'solver': 'nsga2',
        'selection': 'uncertainty',
    }


class MOEAD_EGO(MOBO):
    '''
    MOEA/D-EGO
    '''
    config = {
        'surrogate': 'gp',
        'acquisition': 'ei',
        'solver': 'moead',
        'selection': 'moead',
    }


class ParEGO(MOBO):
    '''
    ParEGO
    '''
    config = {
        'surrogate': 'gp',
        'acquisition': 'ei',
        'solver': 'parego',
        'selection': 'random',
    }


'''
Define new algorithms here
'''


class Custom(MOBO):
    '''
    Totally rely on user arguments to specify each component
    '''
    config = None


def get_algorithm(name):
    '''
    Get class of algorithm by name
    '''
    algo = {
        'dgemo': DGEMO,
        'tsemo': TSEMO,
        'usemo-ei': USEMO_EI,
        'moead-ego': MOEAD_EGO,
        'parego': ParEGO,
        'custom': Custom,
    }
    return algo[name]