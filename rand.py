from random import seed
from torch import manual_seed
from numpy.random import seed as np_seed

def reset_seeds():
    seed(0)
    manual_seed(0)
    np_seed(0)