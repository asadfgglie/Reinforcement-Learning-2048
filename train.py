import os
import pickle
import time

from numpy import argmax
from evolution.genetic.function import genetic_spawn_population, s4_genetic_params_generator, genetic_individual
from evolution.genetic.function import next_generation as genetic_next_generation

from evolution.strategy.function import strategy_spawn_population, s4_strategy_params_generator, strategy_individual
from evolution.strategy.function import next_generation as strategy_next_generation

from evolution.tool import evaluate_population


def genetic_train(store_dir: str, generation: int = 20, pop_size: int = 100, mutation_rate: float = 0.05, mutate_prob: float = 0.01, tournament_rate: float = 0.5) -> None:
    pop_fits = []

    assert pop_size > 2, 'Too small population'
    t = time.localtime()
    result = time.strftime("%Y_%m_%d", t)

    pop = genetic_spawn_population(pop_size, s4_genetic_params_generator, genetic_individual)

    if not os.path.isdir(f'./{store_dir}'):
        os.mkdir(f'./{store_dir}')
    if not os.path.isdir(f'./{store_dir}/{result}'):
        os.mkdir(f'./{store_dir}/{result}')

    path = f'./{store_dir}/{result}'

    with open(os.path.join(path, f'hyperparameter.txt'), 'w') as f:
        f.write(f'generation: {generation}\n')
        f.write(f'pop_size: {pop_size}\n')
        f.write(f'mutation_rate: {mutation_rate}\n')
        f.write(f'mutate_prob: {mutate_prob}\n')
        f.write(f'tournament_rate: {tournament_rate}\n')
        f.write(f'params_generator: {s4_genetic_params_generator}\n')
        f.write(f'model: {genetic_individual}\n')

    i = 0

    def save_model() -> None:
        with open(os.path.join(path, f'generation_{i}.pickle'), 'wb') as f:
            pickle.dump(pop, f)

    try:
        print('Generation 0')
        pop = evaluate_population(pop)
        print('\tAvg fit:', pop.avg_fitness)
        pop_fits.append(pop.avg_fitness)
        save_model()

        for i in range(1, generation):
            pop = genetic_next_generation(pop, mutation_rate, tournament_rate, mutate_prob)

            print(f'\nGeneration {i}')
            pop = evaluate_population(pop)
            print('\tAvg fit:', pop.avg_fitness)
            pop_fits.append(pop.avg_fitness)
            save_model()

    finally:
        with open(os.path.join(path, f'fitness.pickle'), 'wb') as f:
            pickle.dump(pop_fits, f)
        with open(os.path.join(path, f'hyperparameter.txt'), 'a') as f:
            f.write('\n')
            f.write(f'actual_generation: {i}\n')
            f.write(f'max_fitness: {pop_fits[argmax(pop_fits)]}\n')
            f.write(f'max_fitness_generation: {argmax(pop_fits)}')
        save_model()

def strategy_train(store_dir: str, generation: int = 20, pop_size: int = 100, learning_rate=0.1) -> None:
    pop_fits = []

    assert pop_size > 2, 'Too small population'
    t = time.localtime()
    result = time.strftime("%Y_%m_%d", t)

    pop = strategy_spawn_population(pop_size, s4_strategy_params_generator, strategy_individual)

    if not os.path.isdir(f'./{store_dir}'):
        os.mkdir(f'./{store_dir}')
    if not os.path.isdir(f'./{store_dir}/{result}'):
        os.mkdir(f'./{store_dir}/{result}')

    path = f'./{store_dir}/{result}'

    with open(os.path.join(path, f'hyperparameter.txt'), 'w') as f:
        f.write(f'generation: {generation}\n')
        f.write(f'pop_size: {pop_size}\n')
        f.write(f'learning_rate: {learning_rate}\n')
        f.write(f'params_generator: {s4_strategy_params_generator}\n')
        f.write(f'model: {strategy_individual}\n')

    i = 0
    best_avg_fitness = 0

    def save_model() -> None:
        with open(os.path.join(path, f'generation_{i}.pickle'), 'wb') as f:
            pickle.dump(pop, f)

    try:
        print('Generation 0')
        pop = evaluate_population(pop)
        print('\tAvg fit:', pop.avg_fitness)
        pop_fits.append(pop.avg_fitness)
        if int(pop.avg_fitness) > int(best_avg_fitness):
            best_avg_fitness = pop.avg_fitness
            save_model()

        for i in range(1, generation):
            pop = strategy_next_generation(pop, s4_strategy_params_generator, learning_rate=learning_rate)

            print(f'\nGeneration {i+1}')
            pop = evaluate_population(pop)
            print('\tAvg fit:', pop.avg_fitness)
            pop_fits.append(pop.avg_fitness)
            if int(pop.avg_fitness) > int(best_avg_fitness):
                best_avg_fitness = pop.avg_fitness
                save_model()

    finally:
        with open(os.path.join(path, f'fitness.pickle'), 'wb') as f:
            pickle.dump(pop_fits, f)
        with open(os.path.join(path, f'hyperparameter.txt'), 'a') as f:
            f.write('\n')
            f.write(f'actual_generation: {i}\n')
            f.write(f'max_fitness: {pop_fits[argmax(pop_fits)]}\n')
            f.write(f'max_fitness_generation: {argmax(pop_fits)}')
        save_model()

if __name__ == '__main__':
    model_dir = 'model'
    genetic_train(model_dir, 400, 500, mutation_rate=0.075)
    # strategy_train(model_dir)
