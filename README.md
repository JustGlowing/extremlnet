ExtrmlNet
----

ExtrmlNet implements a Single Layer Feedforward Neural Network trained with the Extreme Learning Machine algorithm.

Using ExtrmlNet is easy to use as any sklearn model:

```
from extrmlnet import ExtrmlNet
net = ExtrmlNet(20)
x = np.linspace(-10,10,100)
y = np.vstack((np.cos(x),np.sin(x)*2)).T # the net will learn cos(x) and sin(x)*2 at the same time
net.fit(x,y)
yy = net.predict(x)
```

This software has been inspired by the following paper:

   Guang-bin Huang and Qin-yu Zhu and Chee-kheong Siew.
   Extreme learning machine: A new learning scheme of feedforward neural networks.
   In Proceedings of International Joint Conference on Neural Networks. 2006.
