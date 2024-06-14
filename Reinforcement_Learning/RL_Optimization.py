import numpy as np
import tensorflow as tf


# Define the function to optimize
def func(x):
    return -(x - 3) ** 2 + 5


# Parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Epsilon-greedy parameter
episodes = 1000

# Discretize the state space
num_buckets = 10
state_bounds = [-5, 5]
action_space = [-1, 1]
num_actions = len(action_space)


# Deep Q-network
class DQN(tf.keras.Model):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.dense3 = tf.keras.layers.Dense(num_actions)

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return self.dense3(x)


# Initialize DQN
dqn = DQN(num_actions)
optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)


# Convert continuous state to discrete state
def discretize_state(state):
    discretized_state = np.digitize(state, np.linspace(state_bounds[0], state_bounds[1], num_buckets))
    return discretized_state - 1


# Epsilon-greedy action selection
def choose_action(state):
    if np.random.random() < epsilon:
        return np.random.choice(num_actions)
    else:
        state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = dqn(state_tensor)
        return tf.argmax(action_probs[0]).numpy()


# Q-learning algorithm
for episode in range(episodes):
    state = discretize_state(np.random.uniform(-5, 5))
    done = False

    while not done:
        action = choose_action(state)
        next_state = discretize_state(state + action_space[action])
        reward = func(state + action_space[action])

        with tf.GradientTape() as tape:
            state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
            state_tensor = tf.expand_dims(state_tensor, 0)
            next_state_tensor = tf.convert_to_tensor(next_state, dtype=tf.float32)
            next_state_tensor = tf.expand_dims(next_state_tensor, 0)
            action_tensor = tf.convert_to_tensor(action, dtype=tf.int32)
            reward_tensor = tf.convert_to_tensor(reward, dtype=tf.float32)

            q_values = dqn(state_tensor)
            q_value = tf.gather(q_values, action_tensor)

            next_q_values = dqn(next_state_tensor)
            max_next_q_value = tf.reduce_max(next_q_values)

            target_q_value = reward_tensor + gamma * max_next_q_value

            loss = tf.square(q_value - target_q_value)

        grads = tape.gradient(loss, dqn.trainable_variables)
        optimizer.apply_gradients(zip(grads, dqn.trainable_variables))

        state = next_state

        if state == 0 or state == num_buckets - 1:
            done = True

# Find optimal solution
optimal_solution = []
for i in range(num_buckets):
    state_tensor = tf.convert_to_tensor([i], dtype=tf.float32)
    state_tensor = tf.expand_dims(state_tensor, 0)
    action_probs = dqn(state_tensor)
    optimal_action = tf.argmax(action_probs[0]).numpy()
    optimal_solution.append(optimal_action)
optimal_value = [func((i + 0.5) * (state_bounds[1] - state_bounds[0]) / num_buckets + state_bounds[0]) for i in
                 range(num_buckets)]
print("Optimal Solution:", optimal_solution)
print("Optimal Value:", optimal_value)
