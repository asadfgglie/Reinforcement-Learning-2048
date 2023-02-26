import copy
from typing import Optional, Union
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn, Tensor

from evolution.population import Population

params_temp: dict[str, dict[str, Union[torch.Tensor, str, None]]] = {
    'conv1':{
        'w': None, # shape: (out_channels, in_channels, kH, kW)
        'b': None,  # shape: (out_channels)
        'padding': None
    },
    'conv2': {
        'w': None, # shape: (out_channels, in_channels, kH, kW)
        'b': None,  # shape: (out_channels)
        'padding': None
    },
    'layer3': {
        'w': None,
        'b': None
    }
}

def genetic_individual(x: torch.Tensor, params: dict) -> Tensor:
    x.requires_grad = False

    conv1_weights = params['conv1']['w']
    conv1_bias = params['conv1']['b']

    y = F.conv2d(x, conv1_weights, conv1_bias, padding=params['conv1']['padding'])

    y = F.relu(y)

    conv2_weights = params['conv2']['w']
    conv2_bias = params['conv2']['b']

    y = F.conv2d(y, conv2_weights, conv2_bias, padding=params['conv2']['padding'])

    y = F.relu(y)

    y = nn.Flatten(start_dim=0)(y)

    layer3_weights = params['layer3']['w']
    layer3_bias = params['layer3']['b']

    y = F.linear(y, layer3_weights, layer3_bias)

    y = F.log_softmax(y, dim=0)

    return y

def s4_genetic_params_generator(seed: Optional[int] = None) -> dict[str, dict]:
    r"""
    Adapt map size of (4, 4)

    :param seed: Optional, use to set random seed
    :return: random params for `individual()`
    """
    if seed is not None:
        np.random.seed(seed=seed)

    params: dict[str, dict[str, Union[torch.Tensor, str, None]]] = copy.deepcopy(params_temp)

    params['conv1']['w'] = torch.randn(10, 1, 3, 3) / 10
    params['conv1']['b'] = torch.randn(10) / 10
    params['conv1']['padding'] = 'same'

    params['conv2']['w'] = torch.randn(20, 10, 2, 2) / 10
    params['conv2']['b'] = torch.randn(20) / 10
    params['conv2']['padding'] = 'valid'

    params['layer3']['w'] = torch.randn(4, 20 * 3 * 3) / 10
    params['layer3']['b'] = torch.randn(4) / 10

    return params

def genetic_spawn_population(pop_size: int, params_generator, model_func, seed: Optional[int] = None) -> Population:
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
    for _ in range(pop_size):
        pop_list.append({
            'params': params_generator(seed=seed),
            'fitness': None
        })
    return Population(pop_list, model_func)

def recombine_chromosome(p1: dict, p2: dict) -> tuple[dict, dict]:
    r"""
    染色體交換繁殖演算法

    每一層神經網路都視為一條染色體

    在繁殖時，將母代每一層神經網路攤平，隨機切成兩片，

    再交叉組合，作為子代神經網路該層的參數

    :param p1: 母代1
    :param p2: 母代2
    :return: 兩個不同染色體組合子代
    """

    p1 = p1['params']
    p2 = p2['params']

    c1 = copy.deepcopy(p1)
    c2 = copy.deepcopy(p2)

    for params_key in p1:
        for k in p1[params_key]:
            if k == 'padding':
                continue

            original_shape = p1[params_key][k].shape
            p1[params_key][k] = p1[params_key][k].flatten()
            p2[params_key][k] = p2[params_key][k].flatten()
            params_len = p1[params_key][k].shape[0]

            clip = int(np.random.random() * params_len)

            c1[params_key][k] = c1[params_key][k].flatten()
            c1[params_key][k][: clip] = p1[params_key][k][: clip]
            c1[params_key][k][clip: ] = p2[params_key][k][clip: ]
            c1[params_key][k] = c1[params_key][k].reshape(original_shape)

            c2[params_key][k] = c2[params_key][k].flatten()
            c2[params_key][k][: clip] = p2[params_key][k][: clip]
            c2[params_key][k][clip:] = p1[params_key][k][clip:]
            c2[params_key][k] = c2[params_key][k].reshape(original_shape)

            p1[params_key][k] = p1[params_key][k].reshape(original_shape)
            p2[params_key][k] = p2[params_key][k].reshape(original_shape)

            assert c1[params_key][k].shape == c2[params_key][k].shape and c1[params_key][k].shape == original_shape

    return {'params': c1, 'fitness': None}, {'params': c2, 'fitness': None}

def mutate(x: dict, rate: float, prob: float) -> dict:
    r"""
    突變演算法

    對每一層以機率`prob`決定是否突變，再以`rate`決定要突變該層神經網路中有多少百分比參數要突變

    :param x: 要突變的個體
    :param rate: 突變參數比例
    :param prob: 突變機率
    :return: 突變後個體
    """
    x_params = x['params']
    for params_key in x_params:
        for k in x_params[params_key]:
            if k == 'padding':
                continue

            original_shape = x_params[params_key][k].shape
            x_params[params_key][k] = x_params[params_key][k].flatten()
            params_len = x_params[params_key][k].shape[0]

            mutate_num = int(rate * params_len)

            if mutate_num != 0 and np.random.rand() < prob:
                index = np.random.randint(0, params_len, size=(mutate_num, ))

                x_params[params_key][k][index] = torch.randn(mutate_num) / 10.0

            x_params[params_key][k] = x_params[params_key][k].reshape(original_shape)

            assert x_params[params_key][k].shape == original_shape

    x['params'] = x_params

    return x

def next_generation(pop: Population, mutate_rate: float, tournament_rate: float, mutate_prob: float) -> Population:
    new_pop_list = []
    while len(new_pop_list) < len(pop):
        tournament_size = max(int(tournament_rate * len(pop)), 3)
        index = np.random.choice(len(pop), tournament_size, replace=False)
        assert index.shape[0] >= 3, index.shape[0]

        batch = np.array([[i, individual['fitness']] for (i, individual) in enumerate(pop.individuals) if i in index])

        scores = batch[batch[:, 1].argsort()]

        ind1, ind2 = int(scores[-1][0]), int(scores[-2][0])

        p1, p2 = pop.individuals[ind1], pop.individuals[ind2]

        c1, c2 = recombine_chromosome(p1, p2)

        c1, c2 = mutate(c1, mutate_rate, mutate_prob), mutate(c2, mutate_rate, mutate_prob)

        new_pop_list.extend([c1, c2])

    tmp = Population(new_pop_list, pop.model)
    del pop
    return tmp