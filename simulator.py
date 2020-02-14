import numpy as np
from PID_model import model
from matplotlib import pyplot as plt
from physics_model import physics
from heater_model import heater
from noise import make_noise


time: float = 60 * 5  # seconds
lag = 0.1


kp = 1
kd = 10
ki = 0.0000

heater_power = 2
heater_diff_coeff = 0.0002
heater_max_temp = 30

target = 97
t0 = 23.6

lose_coeff = 0.05

NOISING = True

h = heater(heater_power, heater_diff_coeff, heater_max_temp)
m = model(kp, kd, ki, target, lag)
ph = physics(h, m, t0, lag, lose_coeff)


times = []
temps = []

powers = []

for time_index in range(int(time / lag)):
    this_time = time_index * lag
    times.append(this_time)
    t, h_power, lose_power, total_power = ph.step()
    temps.append(t)
    powers.append(lose_power)


if NOISING:
    noise = make_noise(0, time, lag, target_max=3, percent_sigma=0.2, target_num=len(temps))
    print("noised")
    smoothed_xs = np.arange(0, time, (time - 0) / noise.shape[0])[:noise.shape[0]]

    plt.plot(smoothed_xs, noise)
    plt.show()
    temps += noise

plt.plot(times, np.array(temps))
# plt.plot(times, powers)

plt.show()


