import sys
import torch
import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from computer_vision.ResNet import *
# from ResNet import *

# hyperparameters
hidden_size = 256
learning_rate = 3e-5

# Constants
GAMMA = 0.99
num_steps = 1000
max_episodes = 140
load = False


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4, pic_embedding=10):
        super(ActorCritic, self).__init__()

        self.num_actions = num_actions
        self.critic_linear1 = nn.Linear(num_inputs+pic_embedding, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)

        self.actor_linear1 = nn.Linear(num_inputs+pic_embedding, 128)
        self.actor_linear2 = nn.Linear(128, hidden_size)
        self.actor_linear3 = nn.Linear(hidden_size, num_actions)

        self.resnet = ResNet9(num_classes=pic_embedding)

    def forward(self, state, pic):
        # pic = torch.from_numpy(pic).float().repeat(pic.shape[0], pic.shape[1], 1, 1)
        pic = torch.from_numpy(pic).float().reshape(1, 1, pic.shape[0], pic.shape[1])
        pic_state = self.resnet(pic)
        # state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        state = torch.from_numpy(state).float().unsqueeze(0)

        state = torch.cat((state, pic_state), 1)

        value = F.relu(self.critic_linear1(state))
        value = self.critic_linear2(value)

        policy_dist = F.relu(self.actor_linear1(state))
        policy_dist = F.relu(self.actor_linear2(policy_dist))
        policy_dist = F.softmax(self.actor_linear3(policy_dist), dim=1)

        return value, policy_dist


def a2c(env):
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.n

    if load:
        loaded = torch.load('actor_critic_pic.pth')
        actor_critic = loaded['model']
        actor_critic.load_state_dict(loaded['model_state_dict'])
        ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)
        ac_optimizer.load_state_dict(loaded['optimizer_state_dict'])
    else:
        actor_critic = ActorCritic(num_inputs, num_outputs, hidden_size, pic_embedding=10)
        ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)

    all_lengths = []
    average_lengths = []
    all_rewards = []
    # entropy_term = 0

    for episode in range(max_episodes):
        log_probs = []
        entropy_term = 0
        values = []
        rewards = []

        if episode == max_episodes - 1:
            env.print_on_map()
        state = env.reset()
        for steps in range(num_steps):
            value, policy_dist = actor_critic.forward(state[0], state[1])
            value = value.detach().numpy()[0, 0]
            dist = policy_dist.detach().numpy()

            dist = (dist + 0.1) / (np.sum(dist + 0.1))  # todo: delete?????????

            try:
                action = np.random.choice(num_outputs, p=np.squeeze(dist))
            except ValueError:
                action = np.random.randint(4)

            log_prob = torch.log(policy_dist.squeeze(0)[action])
            entropy = -np.sum(np.mean(dist) * np.log(dist))
            new_state, reward, done, _ = env.step(action)

            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            if entropy > 1000:
                print("test")
            entropy_term += entropy
            state = new_state

            if done or steps == num_steps - 1:
                Qval, _ = actor_critic.forward(new_state[0], new_state[1])
                Qval = Qval.detach().numpy()[0, 0]
                all_rewards.append(np.sum(rewards))
                all_lengths.append(steps)
                average_lengths.append(np.mean(all_lengths[-10:]))
                if episode % 10 == 0:
                    sys.stdout.write("episode: {}, reward: {}, total length: {}, average length: {} \n"
                                     .format(episode, np.sum(rewards), steps, average_lengths[-1]))

                    if not load:
                        torch.save({'model': actor_critic, 'model_state_dict': actor_critic.state_dict(),
                                    'optimizer_state_dict': ac_optimizer.state_dict()}, 'actor_critic_pic.pth')

                break

        # compute Q values
        Qvals = np.zeros_like(values)
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + GAMMA * Qval
            Qvals[t] = Qval

        # update actor critic
        values = torch.FloatTensor(values)
        Qvals = torch.FloatTensor(Qvals)
        log_probs = torch.stack(log_probs)

        if log_probs.sum().abs() > 10000 or entropy_term > 10000:
            print('size_error')

        advantage = Qvals - values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

        ac_optimizer.zero_grad()
        ac_loss.backward()
        ac_optimizer.step()

    # Plot results
    smoothed_rewards = pd.Series.rolling(pd.Series(all_rewards), 10).mean()
    smoothed_rewards = [elem for elem in smoothed_rewards]
    plt.plot(all_rewards)
    plt.plot(smoothed_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

    plt.plot(all_lengths)
    plt.plot(average_lengths)
    plt.xlabel('Episode')
    plt.ylabel('Episode length')
    plt.show()

    plt.imshow(env.map, cmap='gray')
    plt.show()

    if not load:
        torch.save({'model': actor_critic, 'model_state_dict': actor_critic.state_dict(),
                    'optimizer_state_dict': ac_optimizer.state_dict()}, 'actor_critic_pic.pth')


class Environment:
    def __init__(self):
        self.map = self.load_map()
        self.state_space = self.map.shape

        # self.reward = None
        self.current_location = np.array([np.random.randint(self.state_space[0]), np.random.randint(self.state_space[1])])
        self.goal_state = np.array([np.random.randint(self.state_space[0]), np.random.randint(self.state_space[1])])
        self.distance_eval, self.camera = self.observe()

        self.state = np.append(self.current_location, self.goal_state)
        self.state = np.append(self.state, self.distance_eval)
        self.state = [self.state, self.camera]

        self.observation_space = self.state[0]  # num_inputs = env.observation_space.shape[0]

        self.action_space = Actions()  # num_outputs = env.action_space.n

        self.print = False

    def load_map(self, filename='/home/matanweks/AirSim/AirSim/PythonClient/computer_vision/airsim_drone/depth_190.png'):
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        big_pic = np.zeros((500, 500))
        start_x = int(big_pic.shape[0]/2 - gray.shape[0]/2)
        stop_x  = int(big_pic.shape[0]/2 + gray.shape[0]/2)
        start_y = int(big_pic.shape[1]/2 - gray.shape[1]/2)
        stop_y  = int(big_pic.shape[1]/2 + gray.shape[1]/2)

        big_pic[start_x:stop_x, start_y:stop_y] = gray
        return big_pic

    def reset(self):
        self.current_location = [np.random.randint(self.state_space[0]-1),
                                 np.random.randint(self.state_space[1]-1)]
        self.goal_state = [np.random.randint(self.state_space[0]-1), np.random.randint(self.state_space[1]-1)]

        while self.map[self.current_location[0], self.current_location[1]] == 0 or \
                self.map[self.goal_state[0]-1, self.goal_state[1]-1] == 0:
            self.current_location = [np.random.randint(self.state_space[0]-1), np.random.randint(self.state_space[1]-1)]
            self.goal_state = [np.random.randint(self.state_space[0]-1), np.random.randint(self.state_space[1]-1)]

        self.current_location = np.array(self.current_location)
        self.goal_state = np.array(self.goal_state)

        self.distance_eval, self.camera = self.observe()

        if self.print:
            print("start" + str(self.current_location))
            print("Goal" + str(self.goal_state))

        self.state = np.append(self.current_location, self.goal_state)
        self.state = np.append(self.state, self.distance_eval)

        self.state = [self.state, self.camera]

        return self.state

    def step(self, action):
        if self.print:
            self.map[self.current_location[0], self.current_location[1]] = 30

        self.current_location += self.action_space.action(action)

        self.distance_eval, self.camera = self.observe()

        new_state = np.append(self.current_location, self.goal_state)
        new_state = np.append(new_state, self.distance_eval)
        new_state = [new_state, self.camera]

        # new_state = np.append(self.current_location, self.camera)
        self.state = new_state
        reward, done = self.calc_reward()

        return new_state, reward, done, None

    def observe(self, view_box=32):
        # view = np.zeros(3+(view_box*2)**2)
        view = np.zeros(3)

        view[0] = np.abs(self.current_location[0]-self.goal_state[0])
        view[1] = np.abs(self.current_location[1]-self.goal_state[1])
        view[2] = np.sqrt(np.mean((self.current_location-self.goal_state)**2))

        min_x = self.current_location[0] - view_box  #  if self.current_location[0] - view_box >= 0 else 0
        min_y = self.current_location[1] - view_box  #  if self.current_location[1] - view_box >= 0 else 0
        max_x = self.current_location[0] + view_box  #  if self.current_location[0] + view_box < 144 else 143
        max_y = self.current_location[1] + view_box  #  if self.current_location[1] + view_box < 255 else 255

        # observe_map = self.map[min_x:max_x, min_y:max_y].reshape(-1, 1)
        observe_map = self.map[min_x:max_x, min_y:max_y]  # .reshape((view_box*2)**2)

        # view[3:] = observe_map

        # return view
        return view, observe_map

    def calc_reward(self):
        if (self.current_location == self.goal_state).all():  # target reached
            return 1000, True
        else:
            # try:
            #     self.map[self.current_location[0], self.current_location[1]]
            # except:
            #     reward = -10
            #     done = True
            #     return reward, done
            if self.map[self.current_location[0], self.current_location[1]] > 0:  # still in free-space
                # reward = 1/self.state[2] * 100
                if self.state[0][4] > 0:  # X dist
                    # reward += 1/self.state[0][4]
                    reward = -self.state[0][4]
                else:
                    reward = 100

                if self.state[0][5] > 0:  # Y dist
                    # reward += 1 / self.state[0][5]
                    reward -= self.state[0][5]
                else:
                    reward += 100

                if self.state[0][6] > 0:  # MSE
                    # reward += 1 / self.state[0][6]
                    reward -= self.state[0][6] / 100
                else:
                    reward += 0.1

                # if reward > 100:
                #     reward = 100

                # if self.state[4] < 250:
                #     reward += 10
                done = False

            else:
                reward = -1
                done = True
            return reward, done

    def print_on_map(self):
        self.print = True


class Actions:
    def __init__(self):
        self.n = 4  # up

    def action(self, a):
        if a == 0:
            return np.array([1, 0])
        if a == 1:
            return np.array([-1, 0])
        if a == 2:
            return np.array([0, 1])
        if a == 3:
            return np.array([0, -1])


if __name__ == "__main__":
    env = Environment()
    # env = gym.make("CartPole-v0")
    # env.step(0)
    a2c(env)
