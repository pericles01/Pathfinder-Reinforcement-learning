import numpy as np
from stable_baselines3 import DQN
import gym_examples
from gym_examples.wrappers import RelativePosition
from stable_baselines3.common.monitor import Monitor
import gym
import os



if __name__ == '__main__':
    env = gym.make('gym_examples/GridWorld-v0', render_mode="human")
    # wrapper
    env = RelativePosition(env)
    env = Monitor(env)
    saved_model = DQN.load(os.path.normpath("tmp/best_model.zip"))
    # test the model
    obs = env.reset()
    while True:
        action, _states = saved_model.predict(obs, deterministic=False)
        if isinstance(action, np.ndarray):
            action = int(np.mean(action))
        print(f"{action}")
        obs, reward, done, info = env.step(action)
        #env.render()
        if done:
            obs = env.reset()
            env.close()