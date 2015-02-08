import sys, os
sys.path.insert(0, os.path.dirname('..'))
from extrmlnet import ExtrmlNet
import numpy as np
import scipy as sp

if __name__ == '__main__':
    x = np.array([[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]])
    y = np.array([1.0,0.0,0.0,1.0])
    net = ExtrmlNet(1) # can't solve this problem with 1 hidden neuron
    net.fit(x,y)
    print np.round(net.predict(x)), ' =', y
    net = ExtrmlNet(4) # with 4 we can
    net.fit(x,y)
    print np.round(net.predict(x)), '!=', y
