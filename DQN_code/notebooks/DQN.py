import tensorflow as tf
import sys
sys.path.append(r"E:\Server_mantain\single_var_DQN\DQN-Timeseries-Anomaly-Detection-master")
from agents.MemoryBuffer import MemoryBuffer
from agents.NeuralNetwork import NeuralNetwork
from environment.Simulator import Simulator
from agents.SlidingWindowAgent import SlidingWindowAgent
from environment.Config import ConfigTimeSeries
from environment.BaseEnvironment import TimeSeriesEnvironment
from environment.WindowStateEnvironment import WindowStateEnvironment


# for subdir, dirs, files in os.walk("../ts_data/A1Benchmark"):
#     for file in files:
#         if file.find('.csv') != -1:


config = ConfigTimeSeries(window=1)

env = WindowStateEnvironment(
    TimeSeriesEnvironment(verbose=True, filename="Test/sample1.csv", config=config))

dqn = NeuralNetwork(input_dim=env.window_size,
                    input_neurons=env.window_size + 1).keras_model

agent = SlidingWindowAgent(dqn=dqn, memory=MemoryBuffer(max=50000, id="sliding_window"), alpha=0.0001,
                           gamma=0.99, epsilon=1,
                           epsilon_end=0.0, epsilon_decay=0.9, fit_epoch=100, action_space=2, batch_size=64)

simulation = Simulator(10, agent, env, 2)
agent.memory.init_memory(env=env)
simulation.run()