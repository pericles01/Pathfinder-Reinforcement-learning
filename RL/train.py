import gym_examples
from gym_examples.wrappers import RelativePosition
import os
import gym
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True
def evaluate(model, num_episodes=100):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    m_env = model.get_env()
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = m_env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = m_env.step(action)
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)

    return mean_episode_reward


if __name__ == '__main__':
    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make('gym_examples/GridWorld-v0', render_mode="human")
    # wrapper
    env = RelativePosition(env)
    env = Monitor(env, log_dir)
    # train the model
    rl_model = DQN(MlpPolicy, env, verbose=0)
    # Evaluate the trained agent
    # mean_reward, std_reward = evaluate_policy(rl_model, Monitor(env, "rl_log"), n_eval_episodes=100)
    # print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    # _ = evaluate(rl_model)
    # rl_model.save("dqn_GridWorld")
    # del rl_model  # remove to demonstrate saving and loading

    # Create the callback: check every 10 steps
    callback = SaveOnBestTrainingRewardCallback(check_freq=5, log_dir=log_dir)
    timesteps = 900
    rl_model.learn(total_timesteps=timesteps, callback=callback, log_interval=4)
    env.close()
    plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "DQN GridWorld-v0")
    plt.show()
    plt.savefig(os.path.join(log_dir, "DQN GridWorld-v0.png"))
    plt.close()


    # saved_model = DQN.load("dqn_GridWorld")
    # # test the model
    # obs = env.reset()
    # while True:
    #     action, _states = saved_model.predict(obs, deterministic=True)
    #     obs, reward, done, info = env.step(action)
    #     env.render()
    #     if done:
    #         obs = env.reset()