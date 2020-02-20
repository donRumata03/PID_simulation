import numpy as np
from PID_model import model
from matplotlib import pyplot as plt
from physics_model import physics
from heater_model import heater
from noise import make_noise
from error_counter import error_function



time: float = 60 * 5  # seconds
lag = 0.1



heater_power = 2
heater_diff_coeff = 0.0002
heater_max_temp = 30

target = 97
t0 = 23.6

lose_coeff = 0.05

NOISING = False



def simulate(kp, kd, ki):
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

    return list(zip(times, temps))

if __name__ == '__main__':
    Kp = 2.822339225054539
    Kd = 30.478187159450126
    Ki = 0

    graph = np.array(simulate(Kp, Kd, Ki))

    plt.plot(graph.T[0], graph.T[1])
    # plt.plot(times, powers)

    err = error_function(graph, target, 5)
    print("error :", err)
    plt.show()