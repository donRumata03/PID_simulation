import math

import numpy as np
from matplotlib import pyplot as plt

filename = "sterzhen_x.txt"

data = [(float(i.split("\t")[0]), float(i.split("\t")[1])) for i in open(filename, "r").read().replace(",", ".").split("\n")]
print(data)

plt.scatter(*zip(*data))
ls = np.linspace(min(list(zip(*data))[0]), max(list(zip(*data))[0]))
print(ls)
plt.plot(ls, [29.1260359505437 + (280 - 29.1260359505437) * math.exp((355.325812248143 - x) * -0.0134820583820244) for x in ls])
plt.show()

print((2 / 0.0134820583820244 ** 2) / 4.45)