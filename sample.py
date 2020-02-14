import matplotlib.pyplot as plt
import numpy as np


def sgn(x):
    return 1 if x > 0 else (0 if x == 0 else -1)


lag = 0.01
length = 2000.0
x = np.arange(0.0, length, lag)

c1 = (np.sin(x/4) * 0.05 + np.sin(x/10)*0.1 + np.sin(x/100) + np.cos(x/1000))
c2 = x.copy(); c2.fill(80)

c = c2


l = [0]

x2 = np.arange(len(l))
ms = []
plt.plot(x, c)

Kp = 1
Ki = 0
Kd = 10

max_speed = 0.01


max_actual_speed = 0

for i in l:
    S = 0.0
    s = 0

    sentensivity = 200.0
    out = 25.6
    C = 5.0
    Elast = c[0] - out
    Speed = 0
    max_power = 0.00001
    outs = []
    ##############################
    # Out = Integral(Integral(U))#
    ##############################

    for t in range(int(length / lag)):
        e = c[t] - out
        S += e
        s += e ** 4
        U = Kp * e + Ki * S * lag + Kd * (e - Elast) / lag
        if U < 0:
            U = max(U, -max_power)
        else:
            U = min(U, max_power)

        Speed += U / sentensivity
        Speed = sgn(Speed) * min(abs(Speed), max_speed)
        max_actual_speed = max(max_actual_speed, Speed)
        out += Speed
        outs.append(out)
        Elast = e

    plt.plot(x, outs + c1 * 10)
    print(abs(s))
    print(max_actual_speed)

plt.show()


