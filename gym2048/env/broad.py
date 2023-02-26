from typing import Any, Literal, Optional

import gym
import numpy as np
import pygame

from warnings import filterwarnings

from gym2048.space.map import Map

filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool8` is a deprecated alias')


class Broad(gym.Env):
    metadata = {
        'info': 'Each value in array means its number in the broad. Zero means no number. '
                'If you want to create a original version, you can set `game_modes = "original", size = 4` when initialize this env. '
                'When use "rgb_array" as `render_mode`, render won\'t return score broad, it will only return game broad',
        'game_modes': ['original', None],
        "render_modes": ["human", 'rgb_array', None],
        "render_fps": 4,
        'action_space_dict': {
            0: 'UP',
            1: 'DOWN',
            2: 'LEFT',
            3: 'RIGHT',
            'UP': 0,
            'DOWN': 1,
            'LEFT': 2,
            'RIGHT': 3
        }
    }

    reward_range = (0, np.Inf)
    # UP, DOWN, LEFT, RIGHT
    action_space = gym.spaces.Discrete(4)
 # type: ignore
    def __init__(self, size: int = 4, window_size: int = 500, interrupt_count: int = 1000, render_mode: Optional[str] = 'human', is_generate: bool = True, game_mode: Optional[str] = 'original') -> None:
        r"""
        :param size: game broad size
        :param window_size: render frame size
        :param interrupt_count: how many steps the agent make an illegal move to interrupt the game
        :param render_mode: the gym render mode
        :param is_generate: For debug. if `False`, game broad won't generate any new number after making a move
        :param game_mode: if `'original'`, the game broad number generator will follow original game version
        """

        assert render_mode in self.metadata['render_modes'], f'`render_mode` must be in {self.metadata["render_modes"]}, get {render_mode}'
        assert game_mode in self.metadata['game_modes'] ,f'`game_mode` must be in {self.metadata["game_modes"]}, get {game_mode}'

        self._is_generate = is_generate
        self._game_mode = game_mode

        self._size = size
        self._window_size = window_size
        self.observation_space = Map(size)

        self._interrupt_count = interrupt_count

        self._count_reward = 0
        self._is_last_move_illegal = False
        self._illegal_move_count = 0

        self._window = None
        self._clock = None

        self.render_mode = render_mode
        self._map_array = np.zeros([1, size, size], int)

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._count_reward = 0
        self._is_last_move_illegal = False
        self._illegal_move_count = 0

        self.observation_space.seed(seed)
        self.action_space.seed(seed)

        self._map_array.fill(0)

        init_number_count = 2 if self._game_mode == 'original' else 1

        for i in range(init_number_count):
            self._generate_new_value()

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> tuple[dict, int, bool, bool, dict]:
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self.metadata['action_space_dict'][action]

        reward, has_move = self._move(direction)

        if self._is_generate and has_move:
            self._generate_new_value()

        self._count_reward += reward

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, not self.movable() or self._illegal_move_count > self._interrupt_count, False, self._get_info()

    def _generate_new_value(self) -> None:
        index = self.np_random.random(2) * self._size
        while self._map_array[0, int(index[0]), int(index[1])] != 0:
            index = self.np_random.random(2) * self._size

        if self._game_mode == 'original':
            self._map_array[0, int(index[0]), int(index[1])] = 2 if self.np_random.random() < 0.9 else 4
        else:
            max_pow = int(np.log2(np.amax(self._map_array)))
            random_pow = self.np_random.random() * (max_pow/2 - max_pow/4) + max_pow/4
            random_pow = max(1, int(random_pow))
            self._map_array[0, int(index[0]), int(index[1])] = int(2**random_pow)

    def _get_info(self) -> dict:
        return {
            'action_dict': self.metadata['action_space_dict'],
            'max_value': self._map_array.reshape(self._size * self._size)[np.argmax(self._map_array.reshape(self._size * self._size))],
            'count_score': self._count_reward,
            'is_last_move_illegal': self._is_last_move_illegal,
            'illegal_move_count': self._illegal_move_count,
            'interrupted_count': self._interrupt_count
        }

    def render(self):
        return self._render_frame()

    def _render_frame(self):
        if self.render_mode is None:
            return

        pix_square_size = self._window_size / self._size
        pygame.init()
        if self._window is None and self.render_mode == "human":
            pygame.display.init()
            self.window = pygame.display.set_mode((self._window_size, self._window_size + pix_square_size))
            pygame.display.set_caption("gym2048")
        if self._clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self._window_size, self._window_size))
        canvas.fill((255, 255, 255))
        # The size of a single grid square in pixels


        value_text = pygame.font.SysFont('Value', int((self._window_size / self._size) * 0.75))

        for column in range(self._size):
            for row in range(self._size):
                value = self._map_array[0, row, column]
                center = (pix_square_size * (1 + 2 * column) / 2, pix_square_size * (1 + 2 * row) / 2)

                if value != 0:
                    bg = pygame.Surface((pix_square_size, pix_square_size))
                    bg.fill((225, max(0, 180 - value * 3), max(0, 180 - value * 2)))
                    canvas.blit(bg, bg.get_rect(center=center))

                text = value_text.render(str(value), True, (0, 0, 0))
                text_rect = text.get_rect(center=center)

                canvas.blit(text, text_rect)


        # Finally, add some gridlines
        for x in range(self._size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self._window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self._window_size),
                width=3,
            )

        info = pygame.Surface((self._window_size, pix_square_size))
        info.fill('white')
        score = pygame.font.SysFont('Score', int((self._window_size / self._size) * 0.55))
        score_text = score.render('score: ' + str(self._count_reward), True, 'black')
        info.blit(score_text, score_text.get_rect())

        if self.render_mode == 'human':
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(info, info.get_rect(topleft=(0, self._window_size)))
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self) -> None:
        if self._window is not None:
            pygame.display.quit()
            pygame.quit()
        del self

    def _get_obs(self) -> np.ndarray[Any, np.dtype]:
        return self._map_array.copy()

    def _move(self, direction: str) -> tuple[int, bool]:
        reward = 0
        is_move = True
        move_count = 0

        if direction == 'UP':
            while is_move:
                is_move = False
                for column in range(self._size):
                    for row in range(1, self._size):
                        if self._map_array[0, row - 1, column] == 0 and self._map_array[0, row, column] != 0:
                            self._map_array[0, row - 1, column], self._map_array[0, row, column] = self._map_array[0, row, column], 0
                            is_move = True
                            move_count += 1
                        elif self._map_array[0, row - 1, column] == self._map_array[0, row, column] and self._map_array[0, row, column] != 0:
                            reward += self._map_array[0, row - 1, column]
                            self._map_array[0, row - 1, column], self._map_array[0, row, column] = self._map_array[0, row, column] * 2, 0
                            is_move = True
                            move_count += 1

        if direction == 'DOWN':
            while is_move:
                is_move = False
                for column in range(self._size):
                    for row in range(self._size - 2, -1, -1):
                        if self._map_array[0, row + 1, column] == 0 and self._map_array[0, row, column] != 0:
                            self._map_array[0, row + 1, column], self._map_array[0, row, column] = self._map_array[0, row, column], 0
                            is_move = True
                            move_count += 1
                        elif self._map_array[0, row + 1, column] == self._map_array[0, row, column] and self._map_array[0, row, column] != 0:
                            reward += self._map_array[0, row + 1, column]
                            self._map_array[0, row + 1, column], self._map_array[0, row, column] = self._map_array[0, row, column] * 2, 0
                            is_move = True
                            move_count += 1

        if direction == 'RIGHT':
            while is_move:
                is_move = False
                for row in range(self._size):
                    for column in range(1, self._size):
                        if self._map_array[0, row, column - 1] == 0 and self._map_array[0, row, column] != 0:
                            self._map_array[0, row, column - 1], self._map_array[0, row, column] = self._map_array[0, row, column], 0
                            is_move = True
                            move_count += 1
                        elif self._map_array[0, row, column - 1] == self._map_array[0, row, column] and self._map_array[0, row, column] != 0:
                            reward += self._map_array[0, row, column - 1]
                            self._map_array[0, row, column - 1], self._map_array[0, row, column] = self._map_array[0, row, column] * 2, 0
                            is_move = True
                            move_count += 1

        if direction == 'LEFT':
            while is_move:
                is_move = False
                for row in range(self._size):
                    for column in range(self._size - 2, -1, -1):
                        if self._map_array[0, row, column + 1] == 0 and self._map_array[0, row, column] != 0:
                            self._map_array[0, row, column + 1], self._map_array[0, row, column] = self._map_array[0, row, column], 0
                            is_move = True
                            move_count += 1
                        elif self._map_array[0, row, column + 1] == self._map_array[0, row, column] and self._map_array[0, row, column] != 0:
                            reward += self._map_array[0, row, column + 1]
                            self._map_array[0, row, column + 1], self._map_array[0, row, column] = self._map_array[0, row, column] * 2, 0
                            is_move = True
                            move_count += 1

        if move_count == 0:
            if self._is_last_move_illegal:
                self._illegal_move_count += 1
            self._is_last_move_illegal = True
            return 0, False
        else:
            self._illegal_move_count = 0
            self._is_last_move_illegal = False
            return reward, True

    @property
    def state(self):
        return self._get_obs()

    @property
    def count_reward(self) -> int:
        return self._count_reward

    def movable(self) -> bool:
        for column_index in range(self._size):
            for row_index in range(self._size):
                tmp = self._map_array[0, row_index, column_index].item()
                for column in range(column_index, self._size):
                    if column_index == column:
                        continue
                    if self._map_array[0, row_index, column].item() == 0:
                        continue

                    if self._map_array[0, row_index, column].item() == tmp:
                        return True
                    if self._map_array[0, row_index, column].item() != tmp:
                        break
            
                for row in range(row_index, self._size):
                    if row_index == row:
                        continue
                    if self._map_array[0, row, column_index].item() == 0:
                        continue

                    if self._map_array[0, row, column_index].item() == tmp:
                        return True
                    if self._map_array[0, row, column_index].item() != tmp:
                        break
        return not np.all(self._map_array)

    def set_map(self, x):
        self._map_array = x