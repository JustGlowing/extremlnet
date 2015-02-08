import sys, os
sys.path.insert(0, os.path.dirname('..'))
from extrmlnet import ExtrmlNet
import numpy as np
from pylab import plot,show

if __name__ == '__main__':
    net = ExtrmlNet(20)
    x = np.linspace(-10,10,100)
    y = np.vstack((np.cos(x),np.sin(x)*2)).T # the net will learn cos(x) and sin(x)*2 at the same time
    net.fit(x,y)
    yy = net.predict(x)
    plot(x,yy,'-r') # red lines are the output of the network
    plot(x,y,'bo') # blue points are the points that we are trying to fit
    show()
    print np.mean(np.abs(y-yy))