import numpy as np
from matplotlib import pyplot as plt
import json
import random

def normal(x, sigma, mu):
    return (np.exp(-(((x - mu) / sigma)**2)/2)) / (sigma * np.sqrt(2 * np.pi))

def smooth_graph(data : list, percent_sigma : float = 0.5, percent_frame_size : float = 0.5):
    buff = [(0, 0) for _ in range(len(data))] # (Coeff, sum)
    frame_size = int(len(data) * percent_frame_size)
    sigma = percent_sigma * len(data)
    for index, (x, y) in enumerate(data):
        if type(x) == type(1+0j):
            x = x.real
        if type(y) == type(1 + 0j):
            y = y.real
        beg = max(0, int(index - frame_size / 2))
        end = min(len(data), beg + frame_size)
        for next_index in range(beg, end):
            this_coeff = normal(data[next_index][0] - x, sigma, 0)
            buff[next_index] = (buff[next_index][0] + this_coeff, buff[next_index][1] + this_coeff * y)

    # res = [(data[index][0], _y / _x) for index, (_x, _y) in enumerate(buff)]
    res = []

    for i in range(len(buff)):
        res.append((data[i][0], 0 if buff[i][0] == 0 else buff[i][1] / buff[i][0]))


    for index, (x, y) in enumerate(res):
        # print(index, res[index])
        pass

    print()
    return res


def smooth_append_graph(data : list, target_num : int, percent_sigma : float = 0.5):
    min_ = min(np.array(data).T[0])
    max_ = max(np.array(data).T[0])
    step = (max_ - min_) / target_num


    buff = [(0, 0) for _ in range(target_num)] # (Coeff, sum)
    sigma = percent_sigma * len(data)
    for index, (x, y) in enumerate(data):
        if hasattr(x, "real"):
            x = x.real
        if hasattr(y, "real"):
            y = y.real
        for next_index in range(len(buff)):
            next_x = min_ + next_index * step
            this_coeff = normal(next_x - x, sigma, 0)
            buff[next_index] = (buff[next_index][0] + this_coeff, buff[next_index][1] + this_coeff * y)
    res = []

    for i in range(len(buff)):
        res.append((min_ + step * i, 0 if buff[i][0] == 0 else buff[i][1] / buff[i][0]))

    return res


def get_logariphmated_graph_x(data : list):
    return [(np.log(s1), s2) for s1, s2 in data if s1 != 0]

def get_exponentated_graph_x(data : list):
    return [(np.exp(s1), s2) for s1, s2 in data]


def exponentate_graph_x(data : list):
    for i in range(len(data)):
        data[i] = (np.exp(data[i][0]), data[i][1])


def logariphmate_graph_x(data : list):
    for i in range(len(data)):
        data[i] = (np.log(data[i][0]) if data[i][0] != 0 else -100, data[i][1])

def smooth_graph_as_exp(data : list, percent_sigma : float = 0.5, percent_frame_size : float = 0.5):
    exped = get_exponentated_graph_x(data)
    return get_logariphmated_graph_x(smooth_graph(exped, percent_sigma, percent_frame_size))


def smooth_graph_as_log(data : list, percent_sigma : float = 0.5, percent_frame_size : float = 0.5):
    logged = get_logariphmated_graph_x(data)
    return get_exponentated_graph_x(smooth_graph(logged, percent_sigma, percent_frame_size))

def plot_tuple_graph(data : list):
    data_xs = []
    data_ys = []
    for x, y in data:
        data_xs.append(x)
        data_ys.append(y)
    plt.plot(data_xs, data_ys)


def print_as_json(data : object):
    j = json.s(data)
    print(json.dumps(j, indent=4))


def test_smoothing():
    testing_smoothing = [
        (0, 1),
        (2, 0),
        (3, 1),
        (4, 7),
        (5, 3),
        (6, 2)
    ]

    smoothed = smooth_append_graph(testing_smoothing, 1000, 0.05)
    print(smoothed)
    smoothed_xs = []
    smoothed_ys = []

    raw_xs = []
    raw_ys = []
    for s in smoothed:
        smoothed_xs.append(s[0])
        smoothed_ys.append(s[1])

    for s in testing_smoothing:
        raw_xs.append(s[0])
        raw_ys.append(s[1])

    plt.plot(raw_xs, raw_ys)
    plt.plot(smoothed_xs, smoothed_ys)
    plt.show()

if __name__ == "__main__":
    test_smoothing()
