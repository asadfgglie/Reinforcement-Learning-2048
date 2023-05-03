import torch.multiprocessing as mp
from typing import Any

from colorama import Fore
from tqdm import tqdm

import gym
import gym2048
import torch

from evolution.population import Population

def test_model(agent_index: int, agent: dict, model_func) -> dict[str, Any]:
    r"""
    評估個體在環境中的分數

    以"遊戲總得分"為評估標準

    :param agent: 個體
    :param model_func: 族群所使用的模型函數
    :return: 已評估過適應分數的個體
    """
    done = False
    with gym.make('gym2048-v0', disable_env_checker=True, render_mode=None) as env:
        obs, _ = env.reset()

        state = torch.from_numpy(obs).float()
        while not done:
            try:
                action_probs = model_func(state, agent['params'])
                action = torch.distributions.Categorical(probs=action_probs).sample()

                new_obs, r, done, _, info = env.step(action.item())

                state: torch.Tensor = torch.from_numpy(new_obs).float()
            except KeyboardInterrupt:
                print(f'Agent {agent_index}:', flush=True)
                print(info, flush=True)
                print(state.numpy(), flush=True)

        agent['fitness'] = env.env.count_reward

    return agent

def evaluate_population(pop: Population) -> Population:
    r"""
    評估族群的適應度

    使用multiprocessing平行化計算

    :param pop: 一個標準族群
    :return: 所有個體都被評估過的標準族群
    """
    with mp.Pool(mp.cpu_count()) as pool, tqdm(total=len(pop), desc='\tEvaluate', ncols=200, leave=True, bar_format='%s{l_bar}%s{bar}%s| {n_fmt}/{total_fmt} agents done [{elapsed}]'%(Fore.WHITE, Fore.GREEN, Fore.WHITE)) as bar:
        """if need print something in sub process, use `print(x, flush=True)`"""
        individuals = []

        def update_bar(result) -> None:
            individuals.append(result)
            bar.update()


        for i, individual in enumerate(pop.individuals):
            pool.apply_async(test_model, (i, individual, pop.model), callback=update_bar)
        while len(individuals) < len(pop):
            pass

    pop.set_individuals(individuals)

    return pop