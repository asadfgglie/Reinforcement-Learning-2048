import math
from typing import Any, Optional

from numpy import argmax


class Population:

    def __init__(self, individuals: list[dict[str, Any]], model_func) -> None:
        self._individuals = individuals
        for i in range(len(individuals)):
            assert 'params' in individuals[i].keys(), f"key 'params' isn't in {individuals[i]}"
            assert 'fitness' in individuals[i].keys(), f"key 'fitness' isn't in {individuals[i]}"
        self._model_func = model_func

    def set_individuals(self, individuals: list[dict[str, Any]]) -> None:
        self._individuals = individuals

    @property
    def avg_fitness(self) -> Optional[float]:
        try:
            return math.fsum((individual['fitness'] for individual in self._individuals)) / len(self._individuals)
        except TypeError:
            return None

    @property
    def best_individual(self) -> dict[str, Any]:
        self._individuals.sort(reverse=True, key=lambda x: x['fitness'])
        return self._individuals[0]

    @property
    def individuals(self) -> list[dict[str, Any]]:
        return self._individuals

    @property
    def model(self) -> Any:
        return self._model_func

    def __len__(self) -> int:
        return len(self._individuals)
