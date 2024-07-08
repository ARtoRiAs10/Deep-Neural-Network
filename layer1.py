import layer
import numpy as np

x = np.array([[0,1], [0,0], [1,1], [0,1]])


net = layer.Model([
    layer.Linear(32),
    layer.Sigmoid(),
    layer.Linear(16),
    layer.Softmax(),
    layer.Linear(8),
    layer.Tanh(),
    layer.Linear(4),
    layer.Relu(),
    ])

print(net(x))