# About
This project is about to test some reinforcement learning algorithms
on a custom gym environment

## Gym Examples
More about Gym environments and wrappers: [Gym documentation](https://gymnasium.farama.org).

## Installation
- To install the gym environment package locally, run
``pip install -e gym-examples``

- Create a virtual environment and install the depedencies:
``conda env create -f environment.yml``

## Test
- This project uses the [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#using-callback-monitoring-training)
framework to test reinforcement learning techniques.
- To train the reinforcement learning agent, run
``python RL\train.py``
- To test the trained model, run
``python RL\test.py``
