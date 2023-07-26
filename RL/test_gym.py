import gym

env = gym.make("LunarLander-v2")
obs = env.reset()

for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()

env.close()