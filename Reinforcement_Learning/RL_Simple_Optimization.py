import numpy as np


# Define the function to optimize
def func(x):
    return -(x - 3) ** 2 + 5


# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Epsilon-greedy parameter
episodes = 1000

# Discretize the state space
num_buckets = 10
state_bounds = [-5, 5]
action_space = [-1, 1]
num_actions = len(action_space)

# Initialize Q-table
Q = np.zeros((num_buckets, num_actions))


# Convert continuous state to discrete state
def discretize_state(state):
    discretized_state = np.digitize(state, np.linspace(state_bounds[0], state_bounds[1], num_buckets))
    return discretized_state - 1


# Epsilon-greedy action selection
def choose_action(state):
    if np.random.random() < epsilon:
        return np.random.choice(num_actions)
    else:
        return np.argmax(Q[state])


# Q-learning algorithm
for episode in range(episodes):
    state = discretize_state(np.random.uniform(-5, 5))
    done = False

    while not done:
        action = choose_action(state)
        next_state = discretize_state(state + action_space[action])
        reward = func(state + action_space[action])

        # Q-learning update rule
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

        state = next_state

        if state == 0 or state == num_buckets - 1:
            done = True

# Find optimal solution
optimal_solution = np.argmax(Q, axis=1)
optimal_value = [func((i + 0.5) * (state_bounds[1] - state_bounds[0]) / num_buckets + state_bounds[0]) for i in
                 range(num_buckets)]
print("Optimal Solution:", optimal_solution)
print("Optimal Value:", optimal_value)