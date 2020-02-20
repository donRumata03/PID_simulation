from typing import *
import numpy as np

length_err_coeff = 5
median_coeff = 10

up_div_down_err_coeff = 5
max_down_div_coeff = 30

max_temp = 99.7


def error_function(graph, target_temp : float = 97, epsilon : float = 7, debug = False):
    first_index_in_epsilon = -1
    for index, val in enumerate(graph):
        if target_temp - epsilon <= val[1] <= target_temp + epsilon:
            first_index_in_epsilon = index
            break
    dx =  graph[1][0] - graph[0][0]

    length = first_index_in_epsilon * dx if first_index_in_epsilon != -1 else len(graph) * dx
    length_err = abs(length) ** 1.5 * length_err_coeff

    precision_integral = 0
    med_sum = 0
    max_temp_counter = 0
    for index, (x, y) in enumerate(graph[first_index_in_epsilon : ]):
        precision_integral += (y - target_temp) ** 2 if y <= target_temp \
            else ((up_div_down_err_coeff * (y - target_temp)) ** 2
                  if y != max_temp else (max_down_div_coeff * (y - target_temp)) ** 2)

        med_sum += y
    precision_error = precision_integral * dx

    mediana = med_sum / (len(graph) - first_index_in_epsilon)
    median_error = (mediana - target_temp) ** 2 * median_coeff

    if debug:
        print("Length:", length)
        print(f"Max temp was {max_temp_counter} times")
        print("Heating length error:", length_err, "Precision:", precision_error, 'Median error:', median_error)

    return length_err + precision_error + median_error

