import numpy as np
from gym import logger
from gym.spaces.box import Box

class Map(Box):
    def __init__(self, size:int):
        super().__init__(low=0, high=np.Inf, shape=(1, size, size), dtype=int)

    def contains(self, x) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if not isinstance(x, np.ndarray):
            logger.warn("Casting input x to numpy array.")
            try:
                x = np.asarray(x, dtype=self.dtype)
            except (ValueError, TypeError):
                return False

        original_condition = bool(
            np.can_cast(x.dtype, self.dtype)
            and x.shape == self.shape
            and np.all(x >= self.low)
            and np.all(x <= self.high)
        )

        return original_condition and self._is_power_of_2(x)

    def sample(self, mask: None = None) -> np.ndarray:
        tmp = super().sample(mask)
        return np.where(tmp <= 0, tmp, 2**tmp)

    def _is_power_of_2(self, x: np.ndarray):
        if np.all(x == 0):
            return True
        if np.all(x <= 1):
            return False

        print(x & (x - 1))

        return np.all((x & (x - 1)) == 0)
