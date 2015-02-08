import sys, os
sys.path.insert(0, os.path.dirname('..'))
from extrmlnet import ExtrmlNet
import numpy as np
from pylab import plot,show,subplot,title

def twospirals(n_points, noise=.5):
    """
     Returns the two spirals dataset.
    """
    n = np.sqrt(np.random.rand(n_points,1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_points,1)*noise
    d1y = np.sin(n)*n + np.random.rand(n_points,1)*noise
    return np.vstack((np.hstack((d1x,d1y)), np.hstack((-d1x,-d1y)) )), np.hstack((np.zeros(n_points),np.ones(n_points)))

if __name__ == '__main__':
    X,y = twospirals(300, noise=.5) # training set
    net = ExtrmlNet(120)
    net.fit(X,y) # train
    X_test,y_test = twospirals(1000, noise=.8) # new dataset with more more noise
    yy = np.round(net.predict(X_test))

    print 'Error:', np.mean(np.abs(yy-y_test))

    subplot(1,2,1)
    title('training set')
    plot(X[y==0,0],X[y==0,1],'.b')
    plot(X[y==1,0],X[y==1,1],'.r')
    subplot(1,2,2)
    title('Neural Network result')
    plot(X_test[yy==0,0],X_test[yy==0,1],'.b')
    plot(X_test[yy==1,0],X_test[yy==1,1],'.r')
    show()