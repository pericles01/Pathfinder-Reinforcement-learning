import gym_examples
import gymnasium
from gym_examples.wrappers import RelativePosition
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers.legacy import Adam
# from keras.models import Sequential
# from keras.layers import Dense, Flatten
# from keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import ModelIntervalCheckpoint, FileLogger


def build_model(input_shape, m_actions):
    model = Sequential()
    model.add(Dense(24, activation='relu', input_shape=input_shape))
    model.add(Dense(24, activation='relu'))
    model.add(Flatten())
    model.add(Dense(m_actions, activation='linear'))
    return model


def build_agent(a_model, a_actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=a_model, memory=memory, policy=policy,
                   nb_actions=a_actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn


def build_callbacks(env_name):
    checkpoint_weights_filename = 'dqn_' + env_name + "dqn_weights.h5f"
    log_filename = 'dqn_{}_log.json'.format(env_name)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=5000)]
    callbacks.extend([FileLogger(log_filename, interval=100)])
    return callbacks


if __name__ == '__main__':
    env = gymnasium.make('gym_examples/GridWorld-v0', render_mode="human")
    wrapped_env = RelativePosition(env)
    # env.reset()
    # temp = 0
    # for _ in range(3000):
    #     # 0 corresponds to "right", 1 to "up", 2 to "left", 3 to "down"
    #     action = env.action_space.sample()
    #     (observation, reward, done, truncated, info) = env.step(action)
    #     # env.render()
    #     print(f"Distance to goal: {info['distance']}")
    #     print(f"reward: {reward}")
    #     print(" ")
    #     if done:
    #         break
    # env.close()
    states = wrapped_env.observation_space.shape
    actions = wrapped_env.action_space.n

    rl_model = build_model((1, states[0]), actions)
    rl_model.summary()
    dqn = build_agent(rl_model, actions)
    dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
    callbacks = build_callbacks("GridWorld-v0")
    dqn.fit(wrapped_env, nb_steps=300, visualize=False, callbacks=callbacks, verbose=1)
    scores = dqn.test(wrapped_env, nb_episodes=100, visualize=False)
    print(np.mean(scores.history['episode_reward']))
    _ = dqn.test(wrapped_env, nb_episodes=15, visualize=True)
    dqn.save_weights('dqn_weights.h5f', overwrite=True)
