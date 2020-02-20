import random
import numpy as np

from simulator import simulate
from error_counter import error_function
from ga import *

def recursive_gen_all_variants(lasted, res, this_res):
    if not lasted:
        res.append(this_res[:])
        return
    for next_coeff in np.arange(lasted[0][0], lasted[0][1] + lasted[0][2], lasted[0][2]):
        this_res.append(next_coeff)
        recursive_gen_all_variants(lasted[1:], res, this_res)
        del this_res[-1]


def gen_all_variants(coeff_data):
    res = []
    recursive_gen_all_variants(coeff_data, res, [])
    return [tuple(val) for val in res]

def dummy_optimize_coeffs(pid_func, err_func, coeff_data : list): # coeff_data = [(-1, 10, 0.01), (3, 6, 0.01)] -> [1.002, 4.881]
    all_variants = gen_all_variants(coeff_data)
    print(all_variants)
    best_err = 2 ** 40
    best_set = ()

    base_min, base_max = coeff_data[0][:2]

    for coeff_set in all_variants:

        this_graph = pid_func(*coeff_set)
        this_err = err_func(this_graph, 97, 5)
        if this_err < best_err:
            best_err = this_err
            best_set = coeff_set
        if random.random() < 0.03:
            print(coeff_set)
            percent = 100 * (coeff_set[0] - base_min) / (base_max - base_min)
            print(percent, "%")
            print(this_err, best_err, best_set)
            print("____________________________________________________________")
    return best_err, best_set


def ga_optimize(fitness_func, coeff_data : list, pop_size = 10, parents_percent = 0.9, iterations = 100, _error_function = None): # coeff_data : [(1, 10), (0.001, 0.002)...] -> [3, 0.0015, ....]
    num_genes = len(coeff_data)
    number_of_parents = int(parents_percent * pop_size)
    offspring_size = pop_size - number_of_parents
    # Generate population
    pop = []
    for index in range(pop_size):
        this_chromosome = [coeff_data[i][0] + random.random() * (coeff_data[i][1] - coeff_data[i][0]) for i in range(num_genes)]
        pop.append(this_chromosome)

    for generation_num in range(iterations):
        print("Iteration :", generation_num, "of", iterations, "(" + str(100 * generation_num / iterations) + " %)")
        # Calc fitness
        fitness = [fitness_func(i) for i in pop]
        parents = choose_mating_chromosomes(pop, fitness, number_of_parents)
        offsprings = generate_crossover_offsprings(parents, offspring_size)
        pop = parents[:] + offsprings

    best_fitness = fitness_func(pop[0])
    best_solver = pop[0]
    if _error_function is not None:
        best_err = _error_function(pop[0])
    for solution in pop:
        if _error_function is None:
            this_fitness = fitness_func(solution)
            if this_fitness > best_fitness:
                best_fitness = this_fitness
                best_solver = solution
        else:
            this_err = _error_function(solution)
            if this_err < best_err:
                best_err = this_err
                best_solver = solution

    return best_fitness if _error_function is None else best_err, best_solver


def quadratisch_test_error(genome : tuple):
    return (genome[0] - 5) ** 2 + (genome[1] + 1) ** 2

def quadratisch_linear_fitness(genome : tuple):
    return 1 / quadratisch_test_error(genome) # TODO : !!!!!!!!!!!!!!

def quadratisch_exp_fitness(genome : tuple):
    return np.exp(-quadratisch_test_error(genome)) # TODO : !!!!!!!!!!!!!!

def quadratisch_quadro_fitness(genome : tuple):
    return 1 / (quadratisch_test_error(genome) ** 2) # TODO : !!!!!!!!!!!!!!


def test_fitnessing():
    linear_reses1 = [
        ga_optimize(quadratisch_linear_fitness, [(-10, 10), (-10, 10)], _error_function=quadratisch_test_error)[0] for
        _i in range(10)]
    print(linear_reses1)
    print(sum(linear_reses1) / len(linear_reses1))

    exp_reses1 = [
        ga_optimize(quadratisch_exp_fitness, [(-10, 10), (-10, 10)], _error_function=quadratisch_test_error)[0] for _i
        in range(10)]
    print(exp_reses1)
    print(sum(exp_reses1) / len(exp_reses1))

    quadro_reses1 = [
        ga_optimize(quadratisch_quadro_fitness, [(-10, 10), (-10, 10)], _error_function=quadratisch_test_error)[0] for
        _i in range(10)]
    print(quadro_reses1)
    print(sum(quadro_reses1) / len(quadro_reses1))

def coeff_fitness_function(genome : tuple):
    graph = simulate(*genome)
    err = error_function(graph)
    return 1 / err

if __name__ == '__main__':
    best, coeffs = ga_optimize(coeff_fitness_function, [(-1, 20), (-1, 30), (0, 0)])
    print("Best error function:",  1 / best)
    print("Resultive coeffs:", coeffs)

# print(dummy_optimize_coeffs(simulate, error_function, [(100, 200, 1), (0, 0.01, 0.005), (0, 1, 0.1)]))

