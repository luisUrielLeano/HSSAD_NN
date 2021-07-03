import sys
import numpy as np
from nn.app import run 

x= sys.argv[1].split(',')
x = list(map(int, x))
x= [x]
x = np.array(x)

run(x)


