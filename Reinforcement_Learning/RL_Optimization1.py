import gym
import numpy as np
from stable_baselines3 import PPO

# Define a custom environment for the optimization problem
class OptimizationEnv(gym.Env):
    def __init__(self, target_function):
        super(OptimizationEnv, self).__init__()
        self.target_function = target_function
        self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    def reset(self):
        self.state = np.random.uniform(-10, 10, size=(1,))
        return self.state

    def step(self, action):
        self.state = np.clip(self.state + action, -10, 10)
        reward = -self.target_function(self.state)
        done = True  # Single-step episode
        return self.state, reward, done, {}

# Define the target function to optimize
def target_function(x):
    return (x - 3) ** 2  # Example: a simple quadratic function

# Create the environment
env = OptimizationEnv(target_function)

# Train an RL agent on the environment
model = PPO("MlpPolicy", env, verbose=1)
print("PPO")
model.learn(total_timesteps=100)

# Test the trained agent
obs = env.reset()
for _ in range(10):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(f"Action: {action}, Observation: {obs}, Reward: {reward}")
