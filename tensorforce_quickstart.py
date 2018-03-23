from tensorforce import Configuration
from tensorforce.agents import DQNAgent
from tensorforce.execution import Runner
from tensorforce.environments import Environment

import numpy as np
# from tensorforce import util, TensorForceError

max_seq_length = 10
num_vals = 5

class Toy(Environment):
    def __init__(self):
        self.reset()

    def __str__(self):
        return "Toy%s" % str(self.state)

    def close(self):
        pass

    def reset(self):
        self.state = ([0] * max_seq_length * 2)
        self.tape_head = max_seq_length
        self.seq_length = np.random.randint(1, 10)
        self.seq = np.random.randint(1, 10, (self.seq_length))
        self.state[:seq_length] = seq
        return dict(self.state)

    def execute(self, actions):
        reward = 0.0
        print(actions)

        if actions != num_vals + 1:
            self.state[self.tape_head] = actions


        if actions == num_vals + 1:
            if action_type == 'bool' or action_type == 'int':
                correct = np.sum(actions[action_type])
                overall = util.prod(shape)
                self.state[action_type] = ((overall - correct) / overall, correct / overall)
            reward += max(min(self.state[action_type][1], 1.0), 0.0)
        else:


        return dict(self.state), terminal, reward

    @property
    def states(self):
        return dict(shape=(max_seq_length * 2,), type='int')

    @property
    def actions(self):
        return dict(type='int', num_actions=num_vals + 1)



# The agent is configured with a single configuration object
config = Configuration(
    memory=dict(
        type='replay',
        capacity=1000
    ),
    batch_size=8,
    first_update=100,
    target_sync_frequency=10
)

# Network is an ordered list of layers
network_spec = [dict(type='dense', size=32), dict(type='dense', size=32)]

environment = Toy()

agent = DQNAgent(
    states_spec=environment.states,
    actions_spec=environment.actions,
    network_spec=network_spec,
    config=config
)

runner = Runner(agent=agent, environment=environment)

def episode_finished(runner):
    if runner.episode % 100 == 0:
        print(sum(runner.episode_rewards[-100:]) / 100)
    return runner.episode < 100 \
        or not all(reward >= 1.0 for reward in runner.episode_rewards[-100:])

runner.run(episodes=1000, episode_finished=episode_finished)