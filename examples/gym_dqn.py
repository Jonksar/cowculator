import numpy as np

# https://github.com/openai/gym
import gym

# https://github.com/fchollet/keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D
from keras.optimizers import RMSprop

# https://github.com/matthiasplappert/keras-rl
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


ENV_NAME = 'CartPole-v0'

HLS = 100

# Setting the env up
env = gym.make(ENV_NAME)
n_action = env.action_space.n

# Build the neural network
model = Sequential()

model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add((Dense(HLS)))
model.add(Activation('relu'))
model.add((Dense(HLS)))
model.add(Activation('relu'))
model.add((Dense(HLS)))
model.add(Activation('relu'))
model.add(Dense(n_action))
model.add(Activation('linear'))

print model.summary()  # How did it go? * ( ' ^')*

# Configure the RL agent
memory = SequentialMemory(limit=50000)
policy = BoltzmannQPolicy()

dqn = DQNAgent(model=model, nb_actions=n_action, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(optimizer=RMSprop(), metrics=['mae'])

# Training
dqn.fit(env, nb_steps=10000, verbose=2, visualize=False)

# Visualize testing
dqn.test(env, nb_episodes=5, visualize=True)
