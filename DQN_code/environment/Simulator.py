# custom modules
from resources import Utils as utils
from resources.Plots import plot_actions,plot_learn,plot_reward,evaluation_func



class Simulator:
    """
    This class is used to train and to test the agent in its environment
    """

    def __init__(self, max_episodes, agent, environment, update_steps):
        """
        Initialize the Simulator with parameters

        :param max_episodes: How many episodes we want to learn, the last episode is used for evaluation
        :param agent: the agent which should be trained
        :param environment: the environment to evaluate and train in
        :param update_steps: the update steps for the Target Q-Network of the Agent
        """
        self.max_episodes = max_episodes
        self.episode = 0
        self.agent = agent
        self.env = environment
        self.update_steps = update_steps

        # information variables
        self.training_scores = []
        self.test_rewards = []
        self.test_actions = []

    def run(self):
        """
        This method is for scheduling training before testing
        :return: True if finished
        """
        while True:
            start = utils.start_timer()
            start_testing = self.__can_test()
            if not start_testing:
                info = self.__training_iteration()
                print("Training episode {} took {} seconds {}".format(self.episode, utils.get_duration(start), info))
                self.__next__()
            if start_testing:
                self.__testing_iteration()
                print("Testing episode {} took {} seconds".format(self.episode, utils.get_duration(start)))
                break
            self.agent.anneal_epsilon()
        plot_actions(self.test_actions[0], getattr(self.env, "timeseries_labeled"))
        plot_learn(self.training_scores)
        evaluation_func(self.test_actions[0], getattr(self.env, "timeseries_labeled"))

        return True

    def __training_iteration(self):
        """
        One training iteration is through the complete timeseries, maybe this needs to be changed for
        bigger timeseries datasets.

        :return: Information of the training episode, if update episode or normal episode
        """
        rewards = 0
        state = self.env.reset()
        for idx in range(len(
                self.env)):
            action = self.agent.action(state)
            state, action, reward, nstate, done = self.env.step(action)
            rewards += reward
            self.agent.memory.store(state, action, reward, nstate, done)
            state = nstate
            if done:
                self.training_scores.append(rewards)
                break

        if len(self.agent.memory) > self.agent.batch_size:
            self.agent.experience_replay()

        # Target Model Update
        if self.episode % self.update_steps == 0:
            self.agent.update_target_model()
            return "Update Target Model"
        return ""

    def __testing_iteration(self):
        """
        The testing iteration with greedy actions only.
        """
        rewards = 0
        actions = []
        state = self.env.reset()
        self.agent.epsilon = 0
        for idx in range(len(
                self.env)):
            action = self.agent.action(state)
            actions.append(action)
            state, action, reward, nstate, done = self.env.step(action)
            rewards += reward
            state = nstate
            if done:
                actions.append(action)
                self.test_rewards.append(rewards)
                self.test_actions.append(actions)
                break

    def __can_test(self):
        """
        :return: True if last episode, False before
        """
        if self.episode >= self.max_episodes:
            return True
        return False

    def __next__(self):
        # increment episode counter
        self.episode += 1
