import time
import sys
import torch
# import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from ResNet import *
from utils import *

# TODO make hyperparameters as a json_config file
hidden_size = 256
learning_rate = 3e-5

# Constants
GAMMA = 0.99
num_steps = 1000
max_episodes = 1


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        super(ActorCritic, self).__init__()

        self.num_actions = num_actions
        self.num_actions = num_actions
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)

        self.actor_linear1 = nn.Linear(num_inputs, 128)
        self.actor_linear2 = nn.Linear(128, hidden_size)
        self.actor_linear3 = nn.Linear(hidden_size, num_actions)

        self.resnet = ResNet18(num_classes=10)

    def forward(self, state):
        state = np.array(state)
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))

        value = F.relu(self.critic_linear1(state))
        value = self.critic_linear2(value)

        policy_dist = F.relu(self.actor_linear1(state))
        policy_dist = F.relu(self.actor_linear2(policy_dist))
        policy_dist = F.softmax(self.actor_linear3(policy_dist), dim=1)

        return value, policy_dist


class Trainer:
    def __init__(self, env, params=None):
        self.env = env
        self.actor_critic = None
        self.optimizer = None
        self.episode = 0

    def load_a2c(self, load=False):
        num_inputs = len(self.env.state)
        num_outputs = self.env.action_space.n

        if load:
            loaded = torch.load('guidance_model.pth')
            actor_critic = loaded['model']
            actor_critic.load_state_dict(loaded['model_state_dict'])
            ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)
            ac_optimizer.load_state_dict(loaded['optimizer_state_dict'])
            self.episode = loaded['episode'] + 1
        else:
            actor_critic = ActorCritic(num_inputs, num_outputs, hidden_size)
            ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)

        self.actor_critic, self.optimizer = actor_critic, ac_optimizer

        return actor_critic, ac_optimizer

    def train(self, episode, last_state, entropy_term):
        Qval_final, _ = self.actor_critic.forward(last_state)
        Qval_final = Qval_final.detach().numpy()[0, 0]
        Qvals = episode.compute_Qvals(Qval_final)

        # update actor critic
        values = torch.FloatTensor(episode.values)
        Qvals = torch.FloatTensor(Qvals)
        log_probs = torch.stack(episode.log_probs)

        advantage = Qvals - values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

        self.optimizer.zero_grad()
        ac_loss.backward()
        self.optimizer.step()

    def save_model(self, filename='guidance_model.pth'):
        torch.save({'model': self.actor_critic,
                    'model_state_dict': self.actor_critic.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'episode': self.episode}, filename)


class Actions:
    def __init__(self, n):
        self.n = n

    def action(self, x):
        return x/10   # if n=10, action # 2 would be 0.2 (throttle value)


class Environment:
    def __init__(self, init_pose, drone, car, planner, goals, n_actions=10):

        self.reward = 0
        self.state = None
        self.action_space = Actions(n_actions)
        self.drone = drone
        self.drone_current_position = init_pose
        self.car = car
        self.car_current_position = init_pose
        self.planner = planner
        self.goals = goals
        self.current_goal = self.goals.pop(0)
        self.drone.current_goal = self.current_goal
        self.dt = 0

    def reset(self, pose, yaw):
        self.drone.move(pose, yaw)
        self.car.move(pose, yaw)

        self.state = self.get_state()
        return self.state

    def step(self, action, delay):

        self.drone.set_speed(action)
        self.dt += 1
        time.sleep(delay)
        self.state = self.get_state()

        if self.drone.dist(self.current_goal) < 1:

            self.current_goal = self.goals.pop(0)
            self.drone.current_goal = self.current_goal
            done = True if not len(self.goals) else False  # will be true if finished all milestones
            reward = self.calc_reward(done, milestone=True)
            print('Flying to next goal:', self.current_goal)

        else:
            done = False
            reward = self.calc_reward(done, milestone=False)

        return self.state, reward, done, None

    def calc_reward(self, is_done, milestone):

        reward = -self.dt

        reward -= (self.state[0] + self.state[1]/10 +  self.state[2])  # Penalty for relative metrics
        reward += self.state[3]/10  # reward for drone speed

        if not self.state[-1]:  # No line-of-sight penalty
            reward -= 50
        if milestone:           # reached a milestone reward
            reward += 100
        if is_done:
            reward += 200

        return reward

    def get_state(self):
        state_vec, state_dict = get_state_vector(self.drone, self.car, self.planner)
        self.drone.current_pose = state_dict['drone']['pose']
        self.car.current_pose = state_dict['car']['pose']

        return state_vec


class Episode:
    def __init__(self, uuid, logger_name):
        self.uuid = uuid
        self.timestamp = time.time()
        self.experiences = []
        self.logger = logger_name
        self.rewards = []
        self.values = []
        self.log_probs = []

    def add_experience(self, experience):
        self.experiences.append(experience)
        self.rewards.append(experience.reward)
        self.values.append(experience.value)
        self.log_probs.append(experience.log_probs)
        self.log(experience)

    def log(self, experience):
        self.logger.info(str(experience.as_dict()))

    def __len__(self):
        return len(self.experiences)

    def compute_Qvals(self, Qval, GAMMA=0.99):
        Qvals = np.zeros_like(self.values)
        for t in reversed(range(len(self.rewards))):
            Qval = self.rewards[t] + GAMMA * Qval
            Qvals[t] = Qval
        return Qvals

    def export(self):
        # save as numpy/pytorch/...
        pass


class Experience:
    def __init__(self, state, action, reward, next_state, value, log_probs, episode):
        self.episode = episode
        self.state = state   # [
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.value = value
        self.log_probs = log_probs

    def as_dict(self):
        return {'episode': self.episode,
                'state': self.state,
                'action': self.action,
                'reward': self.reward,
                'next_state': self.next_state}


# def a2c(env):
#     num_inputs = env.observation_space.shape[0]
#     num_outputs = env.action_space.n
#
#     load = False
#     if load:
#         loaded = torch.load('actor_critic.pth')
#         actor_critic = loaded['model']
#         actor_critic.load_state_dict(loaded['model_state_dict'])
#         ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)
#         ac_optimizer.load_state_dict(loaded['optimizer_state_dict'])
#     else:
#         actor_critic = ActorCritic(num_inputs, num_outputs, hidden_size)
#         ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)
#
#     all_lengths = []
#     average_lengths = []
#     all_rewards = []
#     # entropy_term = 0
#
#     for episode in range(max_episodes):
#         log_probs = []
#         entropy_term = 0
#         values = []
#         rewards = []
#
#         if episode == max_episodes - 1:
#             env.print_on_map()
#         state = env.reset()
#         for steps in range(num_steps):
#             value, policy_dist = actor_critic.forward(state)
#             value = value.detach().numpy()[0, 0]
#             dist = policy_dist.detach().numpy()
#
#             dist = (dist + 0.1) / (np.sum(dist + 0.1))  # todo: delete?????????
#
#             try:
#                 action = np.random.choice(num_outputs, p=np.squeeze(dist))
#             except ValueError:
#                 action = np.random.randint(4)
#
#             log_prob = torch.log(policy_dist.squeeze(0)[action])
#             entropy = -np.sum(np.mean(dist) * np.log(dist))
#             new_state, reward, done, _ = env.step(action)
#
#             rewards.append(reward)
#             values.append(value)
#             log_probs.append(log_prob)
#             if entropy > 1000:
#                 print("test")
#             entropy_term += entropy
#             state = new_state
#
#             if done or steps == num_steps - 1:
#                 Qval, _ = actor_critic.forward(new_state)
#                 Qval = Qval.detach().numpy()[0, 0]
#                 all_rewards.append(np.sum(rewards))
#                 all_lengths.append(steps)
#                 average_lengths.append(np.mean(all_lengths[-10:]))
#                 if episode % 10 == 0:
#                     sys.stdout.write("episode: {}, reward: {}, total length: {}, average length: {} \n"
#                                      .format(episode, np.sum(rewards), steps, average_lengths[-1]))
#                 break
#
#         # compute Q values
#         Qvals = np.zeros_like(values)
#         for t in reversed(range(len(rewards))):
#             Qval = rewards[t] + GAMMA * Qval
#             Qvals[t] = Qval
#
#         # update actor critic
#         values = torch.FloatTensor(values)
#         Qvals = torch.FloatTensor(Qvals)
#         log_probs = torch.stack(log_probs)
#
#         if log_probs.sum().abs() > 10000 or entropy_term > 10000:
#             print('size_error')
#
#         advantage = Qvals - values
#         actor_loss = (-log_probs * advantage).mean()
#         critic_loss = 0.5 * advantage.pow(2).mean()
#         ac_loss = actor_loss + critic_loss + 0.001 * entropy_term
#
#         ac_optimizer.zero_grad()
#         ac_loss.backward()
#         ac_optimizer.step()
#
#     # Plot results
#     smoothed_rewards = pd.Series.rolling(pd.Series(all_rewards), 10).mean()
#     smoothed_rewards = [elem for elem in smoothed_rewards]
#     plt.plot(all_rewards)
#     plt.plot(smoothed_rewards)
#     plt.plot()
#     plt.xlabel('Episode')
#     plt.ylabel('Reward')
#     plt.show()
#
#     plt.plot(all_lengths)
#     plt.plot(average_lengths)
#     plt.xlabel('Episode')
#     plt.ylabel('Episode length')
#     plt.show()
#
#     plt.imshow(env.map, cmap='gray')
#     plt.show()
#
#     if not load:
#         torch.save({'model': actor_critic, 'model_state_dict': actor_critic.state_dict(),
#                     'optimizer_state_dict': ac_optimizer.state_dict()}, 'actor_critic.pth')
