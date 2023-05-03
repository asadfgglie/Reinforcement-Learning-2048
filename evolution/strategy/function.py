from typing import Optional
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from evolution.population import Population

def strategy_individual(x: torch.Tensor, params: Tensor, params_shape: list = [[10, 1, 3, 3], [10], [20, 10, 2, 2], [20], [4, 20 * 3 * 3], [4]]) -> Tensor:
    x.requires_grad = False

    conv1_weights = params[0: np.prod(params_shape[0])].reshape(params_shape[0])
    conv1_bias = params[np.prod(params_shape[0]): np.prod(params_shape[1])].reshape(params_shape[1])

    y = F.conv2d(x, conv1_weights, conv1_bias, padding='same')

    y = F.relu(y)

    conv2_weights = params[np.prod(params_shape[1]): np.prod(params_shape[2])].reshape(params_shape[2])
    conv2_bias = params[np.prod(params_shape[2]): np.prod(params_shape[3])].reshape(params_shape[3])

    y = F.conv2d(y, conv2_weights, conv2_bias, padding='valid')

    y = F.relu(y)

    y = nn.Flatten(start_dim=0)(y)

    layer3_weights = params[np.prod(params_shape[3]): np.prod(params_shape[4])].reshape(params_shape[4])
    layer3_bias = params[np.prod(params_shape[4]): np.prod(params_shape[5])].reshape(params_shape[5])

    y = F.linear(y, layer3_weights, layer3_bias)

    y = F.log_softmax(y, dim=0)

    return y

def s4_strategy_params_generator(seed: Optional[int] = None, params_shape: list = [[10, 1, 3, 3], [10], [20, 10, 2, 2], [20], [4, 20 * 3 * 3], [4]]) -> Tensor:
    r"""
    Adapt map size of (4, 4)

    :param seed: Optional, use to set random seed
    :return: random params for `individual()`
    """
    if seed is not None:
        torch.seed(seed=seed)
    params_len = 0
    for shape in params_shape:
        params_len += int(np.prod(shape))
    params = torch.randn(params_len)

    return params

def strategy_spawn_population(pop_size: int, params_generator, model_func, seed: Optional[int] = None) -> Population:
    r"""
    已指定的參數生成器生成每一個體的參數

    並將其包裝成標準`Population`

    :param pop_size: 群體大小
    :param params_generator: python function, 參數生成器
    :param model_func: python function, 可滿足該參數生成器的模型
    :param seed: random seed
    :return: A standard population
    """
    pop_list = []
    tmp = None
    noise = None
    if seed is not None:
        tmp = params_generator(seed=seed*2)
    else:
        tmp = params_generator(seed=seed)  
    noise = params_generator(seed=seed)

    pop_list.append({
        'params': tmp,
        'fitness': None
    })

    for _ in range(1, pop_size):
        tmp += noise
        pop_list.append({
            'params': tmp,
            'fitness': None
        })
    return Population(pop_list, model_func)

def next_generation(pop: Population, params_generator, learning_rate: float = 0.1, seed: Optional[int] = None) -> Population:
    fitnesses = Tensor([i['fitness'] for i in pop.individuals])
    fitnesses /= Tensor.sum(fitnesses)

    params = pop.individuals[0]['params'] * fitnesses[0]
    for i in range(1, len(pop)):
        params += pop.individuals[i]['params'] * fitnesses[i]
    new_pop_list = [{'params': pop.individuals[0]['params'] + (params - pop.individuals[0]['params']) * learning_rate, 'fitness': None}]

    n = params_generator(seed)
    for _ in range(1, len(pop)):
        new_pop_list.append({'params': n + new_pop_list[-1], 'fitness': None})

    tmp = Population(new_pop_list, pop.model)
    del pop
    return tmp
