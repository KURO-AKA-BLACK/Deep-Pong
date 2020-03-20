from collections import deque
import cProfile
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.autograd as autograd
import math, random
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class QLearner(nn.Module):
    def __init__(self, env, num_frames, batch_size, gamma, replay_buffer):
        super(QLearner, self).__init__()

        self.batch_size = batch_size
        self.gamma = gamma
        self.num_frames = num_frames
        self.replay_buffer = replay_buffer
        self.env = env
        self.input_shape = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n

        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def feature_size(self):
            return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), requires_grad=True)
            # TODO: Given state, you should write code to get the Q value and chosen action
            value, index = torch.max(self.forward(state),1)
            action = int(index)
            return action
            


        else:
            action = random.randrange(self.env.action_space.n)
        return action

    def copy_from(self, target):
        self.load_state_dict(target.state_dict())

        
def compute_td_loss(model, target_model, batch_size, gamma, replay_buffer):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    b_state, b_action, b_reward, b_next_state, b_done = state, action, reward, next_state, done
    state = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)).squeeze(1), requires_grad=True)
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))
    # implement the loss function here
    total_loss = 0.0
    loss = []
    #pr = cProfile.Profile()
    #pr.enable()
    #replay_buffer_list = []
    #for i in range(len(replay_buffer)):
    #    print(replay_buffer.buffer[i][0].shape)
    #    break
    #    replay_buffer_list.append(replay_buffer.buffer[i][0])
    #replay_buffer_list = torch.FloatTensor(replay_buffer_list)

    for i in range(batch_size):
        #print("hello")
        target_value = float(reward[i])
        cnt = 1
        t_next_state = b_next_state[i]
        t_reward = b_reward[i]
        t_done = bool(done[i])
        path_queue = [b_state[i]]
        while((t_reward != 1) and (t_reward != -1)):
            index = -1#replay_buffer_list.index(torch.FloatTensor(t_next_state))
            for j in range(len(replay_buffer)):
                #comparison = replay_buffer.buffer[j][0] == t_next_state
                if (not((replay_buffer.buffer[j][0] != t_next_state).any())):
                    index = j
                    break
            temp = replay_buffer.buffer[index]        
            t_next_state = temp[3]                    
            t_reward = temp[2]
            target_value = target_value + (gamma ** cnt) * temp[2]
            cnt = cnt + 1
            for k in range(len(path_queue)):
                cmpr = path_queue[k] == t_next_state
                if (cmpr.all()):
                    loss.append(-1)
                    t_reward = -1
                    break
            path_queue.append(t_next_state)
            
            
        #print(target_value)
        #print(target_model.size)
        #print(((target_model.forward(state[i]))[0][b_action[i]] - target_value)**2)
        loss.append(((target_model.forward(state[i]))[0][b_action[i]] - target_value)**2)
    #pr.disable()
    #pr.print_stats(sort='time')
    for i in range(len(loss)):
        total_loss = total_loss + loss[i]
    
    return total_loss/batch_size


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # TODO: Randomly sampling data with specific batch size from the buffer
        state, action, reward, next_state, done = [], [], [], [], []

        samples = random.sample(self.buffer, batch_size)
        for i in range(batch_size):
            state.append(samples[i][0])
            action.append(samples[i][1])
            reward.append(samples[i][2])
            next_state.append(samples[i][3])
            done.append(samples[i][4])
        
        
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
