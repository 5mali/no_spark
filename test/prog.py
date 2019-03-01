from datetime import datetime
import os
from os.path import dirname, abspath, join
from os import getcwd
import sys

import random
import string
import pandas as pd
import numpy as np
import torch
import socket


seed_arg = int(sys.argv[1])

seedlist = np.array([161, 314, 228, 271828, 230, 4271031, 5526538, 6610165, 9849252, 34534, 73422, 8765])
seed = seedlist[seed_arg]
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

NAME       = 'MYPROG'
MODELNAME  = NAME + '_' + str(seed) + '.pt'
print("\nMODEL : ", NAME)
print("SEED  : ",seed_arg)
print("HOST  : ",socket.gethostname())
tic = datetime.now()

NO_OF_ITERATIONS = 1000000
for iteration in range(NO_OF_ITERATIONS):
	x = np.random.random(1)
	x = x**2+x**3

print("X = :", x)

toc = datetime.now()

print("START TIME   : ", tic)
print("END TIME     : ", toc)
print("ELAPSED TIME : ", toc-tic)
