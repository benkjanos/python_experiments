import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            target = reward + self.gamma * torch.max(self.model(next_state).detach())
            target_f = self.model(state).detach().numpy()
            target_f[0][action] = target.numpy()
            target_f = torch.FloatTensor(target_f)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            loss = criterion(target_f, self.model(state))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

if __name__ == "__main__":
    state_size = 4
    action_size = 2
    agent = DQNAgent(state_size, action_size)
    done = False
    batch_size = 32

    for e in range(1000):
        state = np.random.rand(1, state_size)
        for time in range(500):
            action = agent.act(state)
            next_state = np.random.rand(1, state_size)
            reward = np.random.rand(1)
            done = False
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, 1000, time, agent.epsilon))
                break
            agent.replay(batch_size)