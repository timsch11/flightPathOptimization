from environment import PlaneNavigationEnv
from rl.algorithms.dqn import DQN
import torch


def main():
    # init model
    model = torch.nn.Sequential(torch.nn.Linear(2, 128), torch.nn.Linear(128, 32), torch.nn.Linear(32, 4))
    mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    dqn = DQN(model=model, actionSpace)

    # initalize environment and get initale state
    env = PlaneNavigationEnv()
    observation, info = env.reset()

    for _ in range(100):
        env.render()
        print(env.action_space)
        action = env.action_space.sample()  # random action for now
        env.action_space.
        observation, reward, terminated, truncated, info = env.step(action)

        print(observation, reward, terminated, truncated, info)
        
        # if destination is reached or scenario is ended
        if terminated or truncated:
            print(f"Episode finished! Reward: {reward}")
            break

    env.close()


if __name__ == '__main__':
    main()