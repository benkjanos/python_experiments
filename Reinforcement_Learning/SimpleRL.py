import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

# Define the environment
# Here we have a 3x3 gridworld
# 0 represents empty cell, 1 represents a wall, and 2 represents the goal
environment = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [0, 2, 0]
])


# Convert state (x, y) to index
def state_to_index(state):
    return state[0] * 3 + state[1]


# Convert index to state (x, y)
def index_to_state(index):
    return (index // 3, index % 3)


# Define the reward function
def reward(state):
    if environment[state[0], state[1]] == 2:
        return 1
    else:
        return 0


# Define the neural network
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Define the epsilon-greedy policy
def epsilon_greedy(net, state, epsilon):
    if np.random.rand() < epsilon:
        return random.choice([0, 1, 2, 3])
    else:
        return torch.argmax(net(state)).item()


# Define the experience replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, state, action, next_state, reward):
        self.buffer.append((state, action, next_state, reward))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return random.sample(self.buffer, len(self.buffer))
        else:
            return random.sample(self.buffer, batch_size)



# Define the Deep Q-learning algorithm
def deep_q_learning(net, target_net, replay_buffer, optimizer, gamma=0.99, batch_size=32, episodes=1000, epsilon=0.1):
    criterion = nn.MSELoss()
    for _ in range(episodes):
        state = torch.eye(9)[0].unsqueeze(0)  # One-hot encode the starting state
        done = False
        while not done:
            action = epsilon_greedy(net, state, epsilon)
            next_state_index = (torch.argmax(state) // 3, torch.argmax(state) % 3)
            next_state = torch.eye(9)[state_to_index(next_state_index)].unsqueeze(0)
            reward_val = reward(next_state_index)
            replay_buffer.push(state, action, next_state, reward_val)

            # Sample from the replay buffer
            batch = replay_buffer.sample(batch_size)
            state_batch, action_batch, next_state_batch, reward_batch = zip(*batch)
            state_batch = torch.cat(state_batch)
            action_batch = torch.tensor(action_batch).unsqueeze(1)
            next_state_batch = torch.cat(next_state_batch)
            reward_batch = torch.tensor(reward_batch).float().unsqueeze(1)

            Q = net(state_batch).gather(1, action_batch)
            Qmax = torch.max(target_net(next_state_batch), 1)[0].unsqueeze(1)
            targetQ = reward_batch + gamma * Qmax
            loss = criterion(Q, targetQ)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            if reward_val == 1:
                done = True


# Define the main function
def main():
    net = DQN(9, 128, 4)
    target_net = DQN(9, 128, 4)
    target_net.load_state_dict(net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    replay_buffer = ReplayBuffer(capacity=1000)

    # Train the network using Deep Q-Learning
    deep_q_learning(net, target_net, replay_buffer, optimizer)

    # Print the learned Q-values
    learned_q_values = net(torch.eye(9)).detach().numpy()
    print("Learned Q-values:")
    print(learned_q_values)

    # Find the optimal policy
    optimal_policy = np.zeros((3, 3), dtype=str)
    for i in range(3):
        for j in range(3):
            if environment[i, j] == 1:
                optimal_policy[i, j] = 'W'  # W represents wall
            elif environment[i, j] == 2:
                optimal_policy[i, j] = 'G'  # G represents goal
            else:
                optimal_policy[i, j] = np.argmax(
                    net(torch.eye(9)[state_to_index((i, j))].unsqueeze(0)).detach().numpy())
    print("\nOptimal Policy:")
    print(optimal_policy)


if __name__ == "__main__":
    main()
