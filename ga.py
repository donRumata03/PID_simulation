import random


def make_choice(russian_roulette_map, val):
    r : int = len(russian_roulette_map)
    l : float = -1
    while r != l + 1:
        mid = int((r + l) / 2)
        if val > russian_roulette_map[mid][1]:
            l = mid
        else:
            r = mid
    return r if r < len(russian_roulette_map) else l


def choose_mating_chromosomes(population, fitness, number):
    russian_roulette_map = []
    norm_coeff = sum(fitness)
    map_pointer = 0
    for index in range(len(population)):
        this_prob = fitness[index] / norm_coeff
        russian_roulette_map.append((map_pointer, map_pointer + this_prob))
        map_pointer += this_prob
    res = []
    for index in range(number):
        res.append(population[make_choice(russian_roulette_map, random.random())])
    return res


def generate_crossover_offsprings(parents, res_size):
    res = []
    for index in range(res_size):
        parent1_idx = random.randint(0, len(parents) - 1)
        parent1 = parents[parent1_idx]
        parent2_idx = random.randint(0, len(parents) - 1)
        while parent2_idx == parent1_idx:
            parent2_idx = random.randint(0, len(parents) - 1)
        parent2 = parents[parent2_idx]

        div_point = int(random.normalvariate(len(parent2) / 2, len(parent2) / 5))
        this_offspring = (parent1[:])[:div_point] + parent2[div_point:]
        res.append(this_offspring)

    return res

# TODO: MUTATION


def mutate_offsprings(offsprings, coeff_range : list, percent_value):
    for offspring in offsprings:
        this_mut_index = random.randint(len(offspring))
        value = percent_value * coeff_range

if __name__ == '__main__':
    """
    res_s = choose_mating_chromosomes([
        ("hello!", False), "Not hello!", object, None],
        [1, 1, 1, 10], 100)

    for i in res_s:
        print(i)
    """
    _parents = [(0, 1, 9), (1, 10, 8), (0, 1, 7), (0, 1, 2), (8, 109, -1)]
    res_s = generate_crossover_offsprings(_parents, 100)

    for i in res_s:
        print(i)

