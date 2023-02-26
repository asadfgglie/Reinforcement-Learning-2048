
import math
from typing import Any

# 為了向後相容舊版程式碼訓練的代理人用
@DeprecationWarning
class Population:

    def __init__(self, individuals: list[dict[str, Any]], model_func):
        self._individuals = individuals
        self._model_func = model_func

    def set_individuals(self, individuals: list[dict[str, Any]]):
        self._individuals = individuals

    @property
    def avg_fitness(self):
        try:
            return math.fsum((individual['fitness'] for individual in self._individuals)) / len(self._individuals)
        except TypeError:
            return None

    @property
    def best_individual(self):
        self._individuals.sort(reverse=True, key=lambda x: x['fitness'])
        return self._individuals[0]

    @property
    def individuals(self):
        return self._individuals

    @property
    def model(self):
        return self._model_func

    def __len__(self):
        return len(self._individuals)
