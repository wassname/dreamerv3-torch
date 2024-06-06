import gymnasium as gym
import numpy as np
from einops import rearrange
import torch.nn.functional as F

from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.play_craftax import CraftaxRenderer
from craftax.craftax.renderer import (
    render_craftax_pixels,
    render_craftax_text,
    inverse_render_craftax_symbolic,
)
from craftax.craftax.constants import Action, Achievement
from craftax.craftax.craftax_state import EnvState

import gymnasium
# from gymnasium.wrappers.jax_to_torch import jax_to_torch
# from gymnasium.wrappers.numpy_to_torch import numpy_to_torch
from gymnasium.wrappers import FrameStack, TimeLimit, FrameStack
import gymnasium.spaces as gym_spaces
from gymnasium.wrappers import TransformObservation

# import jax
import chex
import jax.numpy as jnp
import torch
from jaxtyping import Float, Int, Bool
from torch import Tensor
from typing import Optional, Tuple, Union, Any, Dict
from jax import dlpack as jax_dlpack
from torch.utils import dlpack as torch_dlpack

from envs.gymnax2gymnasium import GymnaxToGymWrapper, GymnaxToVectorGymWrapper

def state2img(state: chex.Array, env_state: EnvState) -> np.ndarray:
    img = inverse_render_craftax_symbolic(state, env_state).astype(np.uint8)
    return np.array(img).astype(np.uint8)


def permute_env(env, prm=[1, 0, 2]):
    os = env.observation_space
    oshape = os.shape
    new_os = gym_spaces.Box(
        low=np.transpose(os.low, prm),
        high=np.transpose(os.high, prm),
        shape=[oshape[i] for i in prm],
        dtype=os.dtype,
    )
    env = TransformObservation(env, lambda x: jnp.transpose(x, prm), obs_space=new_os)
    return env


def jax_to_torch(v) -> torch.Tensor:
    dlpack = jax_dlpack.to_dlpack(v)
    return torch_dlpack.from_dlpack(dlpack)

def numpy_to_torch(v) -> torch.Tensor:

    # for lazyframes
    if hasattr(v, '_frames'): v = np.array(v._frames)
    return torch.from_numpy(v)

def to_torch(v) -> torch.Tensor:
    if isinstance(v, jnp.ndarray):
        if v.dtype=='bool':
            # bool doesn't convert using the jax_to_torch dlpack
            # return torch.from_numpy(v._npy_value.copy())
            return torch.as_tensor(v.tolist())
        return jax_to_torch(v)
    if isinstance(v, np.ndarray):
        return numpy_to_torch(v)
    if isinstance(v, torch.Tensor):
        return v
    else:
        return torch.as_tensor(v)



class CraftaxCompatWrapper(gymnasium.core.Wrapper):
    """
    Misc compat
    - from jax
    """

    def __init__(self, env) -> None:
        super().__init__(env)
        self._env = env.unwrapped._env

    def step(
        self, action: int
    ) -> Tuple[Float[Tensor, "frames odim"], float, bool, bool, Dict]:
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        return (
            numpy_to_torch(next_obs).to(torch.float16), # in symbolic only lighting needs values other than 0 and 1
            to_torch(reward),
            # to_torch(terminated),
            to_torch(truncated | terminated),
            info,
        )

    def reset(self, *args, **kwargs):
        obs, state = self.env.reset(*args, **kwargs)
        return numpy_to_torch(obs).to(torch.float16), state

    def get_action_meanings(self) -> Dict[int, str]:
        return {i.value: s for s, i in Action.__members__.items()}

    @property
    def env_state(self):
        return self.env.unwrapped.env_state


class CraftaxRenderWrapper(gymnasium.core.Wrapper):
    """
    Wrap Gymax (jas gym) to Gym (original gym)
    The main difference is that Gymax needs a rng key for every step and reset
    """

    def __init__(self, env, render_method: Optional[str] = None) -> None:
        super().__init__(env)
        self.render_method = render_method
        if render_method == "play":
            self.renderer = CraftaxRenderer(
                self.env, self.env_params, pixel_render_size=1
            )
        self.renderer = None

    def step(self, *args, **kwargs):
        o = self.env.step(*args, **kwargs)
        if self.renderer is not None:
            self.renderer.update()
        return o

    def reset(self, *args, **kwargs):
        o = self.env.reset(*args, **kwargs)
        if self.renderer is not None:
            self.renderer.update()
        return o

    def render(self, mode="rgb_array"):
        o = self.env.render()
        if self.renderer is not None:
            return self.renderer.render(self.env_state)
        elif self.render_method == "text":
            return render_craftax_text(self.env_state)
        else:
            return render_craftax_pixels(self.env_state, 10)
        return o

    def close(self):
        if self.renderer is not None:
            self.renderer.pygame.quit()
            self.renderer.close()


def create_craftax_env(
    game="Craftax-Symbolic-AutoReset-v1", frame_stack=2, time_limit=None, seed=42, eval=False, num_envs=1
):
    """
    Craftax with
    - frame_stack 4?
    time_limit = 27000

    """
    # see https://github.dev/MichaelTMatthews/Craftax_Baselines/blob/main/ppo_rnn.py
    assert 'AutoReset' in game, f"Only AutoReset games supported, got {game}"
    env = make_craftax_env_from_name(game, auto_reset=True)
    if num_envs > 1:
        # FIXME: naive optimistic resets don't work well with multiple envs see OptimisticResetVecEnvWrapper
        env = GymnaxToVectorGymWrapper(env, seed=seed, num_envs=num_envs)
        raise NotImplementedError("Only num_envs > 1 supported FIXME")
    else:
        env = GymnaxToGymWrapper(env, env.default_params, seed=seed)
    # env = LogWrapper(env)

    # We have to vectorise using jax earlier as there is not framestack wrapepr avaiable for jax
    env = FrameStack(env, frame_stack)
    if num_envs > 1:
        # but then the framestack dim is before the env dim [framestack, batch, obs_dim] so lets swap those
        env = permute_env(env, [1, 0, 2])

    # env.unwrapped.spec = gym.spec(game) # required for AtariPreprocessing
    if not eval and time_limit is not None:
        env = TimeLimit(env, max_episode_steps=time_limit)

    env = CraftaxRenderWrapper(env, render_method=None)
    env = CraftaxCompatWrapper(env)
    return env


def reshape_state(state: Float[Tensor, 'frames state_dim']) -> (Float[Tensor,'frames h w c'], Float[Tensor,'frames inv']):
    """
    reshapes state into map and inv

    https://github.com/MichaelTMatthews/Craftax/blob/main/obs_description.md
    """
    map = rearrange(state[:, :8217], 'frames (h w c) -> frames h w c', h=9, w=11, c=83)
    # now pad from (9,11) to (16,16) so that it's 2*n
    map = F.pad(map, (0, 0, 3, 2, 4, 3))
    map = rearrange(map, 'f h w c -> h w (f c)')
    inventories = rearrange(state[:, 8217:], 'frames c -> (frames c)')
    return map, inventories

class Craftax:
    metadata = {}

    def __init__(self, task, seed=0):

        self._env = create_craftax_env(task, seed=seed)
        # self._achievements = crafter.constants.achievements.copy()
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        frames = self._env.observation_space.shape[0]
        spaces = {
            # "state": gym.spaces.Box(0, 1, (np.prod(self._env.observation_space.shape),), dtype=np.float32),
            "state_map": gym.spaces.Box(0, 1, (16, 16, frames*83), dtype=np.float16),
            "state_inventory": gym.spaces.Box(0, 1, (frames * 51,), dtype=np.float16),
            "image": gym.spaces.Box(0, 255, (130, 110, 3), dtype=np.uint8),
            "is_first": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
            "is_last": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
            "is_terminal": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
            "log_reward": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
        }

        spaces.update(
            {
                f"log_achievement_{k.name.lower()}": gym.spaces.Box(
                    -np.inf, np.inf, (1,), dtype=np.float32
                )
                for k in Achievement
            }
        )
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        action_space = self._env.action_space
        # action_space.discrete = True
        return action_space

    def step(self, action):
        state, reward, done, info = self._env.step(action)

        info2 = {k.replace('Ach','log_ach'):v for k,v in info.items()}

        reward = np.float32(reward)
        state_map, state_inv = reshape_state(state)
        obs = {
            "image": self.get_image(),
            "state": state.flatten(),
            "state_map": state_map,
            "state_inventory": state_inv,
            "is_first": False,
            "is_last": done,
            "is_terminal": info["discount"] == 0,
            **info2,
        }
        return obs, reward, done, info

    def render(self):
        return self._env.render()
    
    def get_image(self):
        # it looks like we need an image in the obs, even if it's not used, so that we can record videos?
        image = render_craftax_pixels(self._env.env_state, 10).astype(np.uint8)
        image = np.array(image).astype(np.uint8)
        return image
    
    def reset(self, seed=None, options=None):
        state, info = self._env.reset()
        state_map, state_inv = reshape_state(state)
        obs = {
            "image": self.get_image(),
            "state": state.flatten(),
            "state_map": state_map,
            "state_inventory": state_inv,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
        }
        return obs
