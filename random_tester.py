from graphic_smoother import *
from matplotlib import pyplot as plt

plt.plot(*zip(*smooth_graph([(i, random.random()) for i in range(1000)], 0.5, 0.5)))
plt.show()